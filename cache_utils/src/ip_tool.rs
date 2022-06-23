use crate::mmap::MMappedMemory;
use bitvec::prelude::*;
use lazy_static::lazy_static;
use std::collections::LinkedList;
use std::ptr::copy_nonoverlapping;
use std::sync::Mutex;
use std::vec::Vec;

struct WXRange {
    start: usize,
    end: usize,     // points to the last valid byte
    bitmap: BitVec, // fixme bit vector
    pages: Vec<MMappedMemory<u8>>,
}

struct WXAllocator {
    ranges: LinkedList<WXRange>,
    // Possible improvement : a dedicated data structure, with optimised lookup of which range
    // contains the right address, plus reasonably easy ability to merge nodes
}

impl WXAllocator {
    fn new() -> Self {
        WXAllocator {
            ranges: LinkedList::<WXRange>::new(),
        }
    }
}

pub struct FunctionTemplate {
    start: unsafe extern "C" fn(*const u8) -> u64,
    ip: *const u8,
    end: *const u8,
}

// Note those fields should not be public
// We need a way to also take care of non allocated functions.
#[derive(Debug)]
pub struct Function {
    pub fun: unsafe extern "C" fn(*const u8) -> u64,
    pub ip: *const u8,
    pub end: *const u8,
    pub size: usize,
}
lazy_static! {
    static ref wx_allocator: Mutex<WXAllocator> = Mutex::new(WXAllocator::new());
}
pub const TIMED_MACCESS: FunctionTemplate = FunctionTemplate {
    start: timed_maccess_template,
    ip: timed_maccess_template_ip as *const u8,
    end: timed_maccess_template_end as *const u8,
};

pub const TIMED_CLFLUSH: FunctionTemplate = FunctionTemplate {
    start: timed_clflush_template,
    ip: timed_clflush_template_ip as *const u8,
    end: timed_clflush_template_end as *const u8,
};

pub const TIMED_NOP: FunctionTemplate = FunctionTemplate {
    start: timed_nop_template,
    ip: timed_nop_template_ip as *const u8,
    end: timed_nop_template_end as *const u8,
};

impl WXRange {
    unsafe fn allocate(
        &mut self,
        align: usize,
        offset: usize,
        length: usize,
        mask: usize,
        round_mask: usize,
    ) -> Result<*mut u8, ()> {
        // In each range, we want to find base = 2^a * k such that start <= base + offset < start + 2^a
        // This can be done with k = ceil(start - align / 2^a).
        // 2^a * k can likely be computed with some clever bit tricks.
        // \o/
        let start = self.start;
        println!(
            "offset: {:x}, align: {:x}, start: {:x}, mask {:x}, round_mask {:x}",
            offset, align, start, mask, round_mask
        );

        let mut candidate = ((start - offset + mask) & round_mask) + offset;
        assert_eq!(candidate & mask, offset);
        assert!(candidate >= start);
        while candidate + length <= self.end {
            let bit_range = &mut self.bitmap[(candidate - start)..(candidate - start + length)];
            if !bit_range.any() {
                bit_range.set_all(true);
                return Ok(candidate as *mut u8);
            }
            candidate += align;
        }
        Err(())
    }

    unsafe fn deallocate(&mut self, p: *const u8, size: usize) {
        let offset = p as usize - self.start;
        if !self.bitmap[offset..(offset + size)].all() {
            panic!("deallocating invalid data");
        }
        self.bitmap[offset..(offset + size)].set_all(false);
    }
}

impl WXAllocator {
    pub unsafe fn allocate(
        &mut self,
        align: usize,
        offset: usize,
        length: usize,
    ) -> Result<*mut u8, ()> {
        if align.count_ones() != 1 && offset < align {
            return Err(()); // FIXME Error type.
        }
        let mask = align - 1;
        let round_mask = !mask;
        loop {
            for range in self.ranges.iter_mut() {
                if let Ok(p) = unsafe { range.allocate(align, offset, length, mask, round_mask) } {
                    return Ok(p);
                }
            }
            const PAGE_SIZE: usize = 1 << 12;
            let size = (length + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            let new_page = MMappedMemory::try_new(size, false, true, |size| 0xcc as u8);
            match new_page {
                Err(_) => return Err(()),
                Ok(new_page) => {
                    let start = &new_page.slice()[0] as *const u8 as usize;
                    let end = start + new_page.len() - 1;
                    let mut cursor = self.ranges.cursor_front_mut();
                    loop {
                        if let Some(current) = cursor.current() {
                            if current.end == start {
                                current.end = end;
                                current.bitmap.append(&mut bitvec![0; end - start]);
                                current.pages.push(new_page);
                                break;
                            }
                            if current.start < start {
                                cursor.move_next()
                            } else {
                                if end == current.start {
                                    current.start = start;
                                    let mut bitmap = bitvec![0; end - start];
                                    bitmap.append(&mut current.bitmap);
                                    current.bitmap = bitmap;
                                    let mut pages = vec![new_page];
                                    pages.append(&mut current.pages);
                                    current.pages = pages;
                                    break;
                                } else {
                                    cursor.insert_before(WXRange {
                                        start,
                                        end,
                                        bitmap: bitvec![0;end-start],
                                        pages: vec![new_page],
                                    });
                                    break;
                                }
                            }
                        } else {
                            cursor.insert_before(WXRange {
                                start,
                                end,
                                bitmap: bitvec![0;end-start],
                                pages: vec![new_page],
                            });
                            break;
                        }
                    }
                }
            }
        }
    }

    pub unsafe fn deallocate(&mut self, p: *const u8, size: usize) {
        let start = p as usize;
        for range in self.ranges.iter_mut() {
            if range.start <= start && start + size - 1 <= range.end {
                unsafe { range.deallocate(p, size) };
            }
        }
    }
}

impl Function {
    pub fn try_new(
        align: usize,
        offset: usize,
        template: FunctionTemplate,
    ) -> Result<Function, ()> {
        // find suitable target
        let mut allocator = wx_allocator.lock().unwrap();
        if align.count_ones() != 1 && offset < align {
            return Err(()); // FIXME Error type.
        }
        let mask = align - 1;
        let real_offset = (offset
            .wrapping_add(template.start as usize)
            .wrapping_sub(template.ip as usize))
            & mask;
        let length = (template.end as usize) - (template.start as usize);

        let p = unsafe { allocator.allocate(align, real_offset, length) }?;
        unsafe { copy_nonoverlapping(template.start as *const u8, p, length) };
        unsafe { std::arch::x86_64::__cpuid(0) };
        let res = Function {
            fun: unsafe {
                std::mem::transmute::<*mut u8, unsafe extern "C" fn(*const u8) -> u64>(p)
            },
            ip: unsafe { p.add(template.ip as usize - template.start as usize) },
            end: unsafe { p.add(length) },
            size: length,
        };
        Ok(res)
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        // Find the correct range, and deallocate all the bits
        let p = self.fun as *mut u8;
        unsafe { std::ptr::write_bytes(p, 0xcc, self.size) };
        let mut allocator = wx_allocator.lock().unwrap();
        unsafe { allocator.deallocate(self.fun as *const u8, self.size) };
    }
}

global_asm!(
    ".global timed_maccess_template",
    "timed_maccess_template:",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "lfence",
    ".global timed_maccess_template_ip",
    "timed_maccess_template_ip:",
    "mov rdi, [rdi]",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "lfence",
    "sub rax, rsi",
    "ret",
    ".global timed_maccess_template_end",
    "timed_maccess_template_end:",
    "nop",
    ".global timed_clflush_template",
    "timed_clflush_template:",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "lfence",
    ".global timed_clflush_template_ip",
    "timed_clflush_template_ip:",
    "clflush [rdi]",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "lfence",
    "sub rax, rsi",
    "ret",
    ".global timed_clflush_template_end",
    "timed_clflush_template_end:",
    "nop",
    ".global timed_nop_template",
    "timed_nop_template:",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "lfence",
    ".global timed_nop_template_ip",
    "timed_nop_template_ip:",
    "nop",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "lfence",
    "sub rax, rsi",
    "ret",
    ".global timed_nop_template_end",
    "timed_nop_template_end:",
    "nop",
);

extern "C" {
    fn timed_maccess_template(pointer: *const u8) -> u64;
    fn timed_maccess_template_ip();
    fn timed_maccess_template_end();
    fn timed_clflush_template(pointer: *const u8) -> u64;
    fn timed_clflush_template_ip();
    fn timed_clflush_template_end();
    fn timed_nop_template(pointer: *const u8) -> u64;
    fn timed_nop_template_ip();
    fn timed_nop_template_end();
}

pub fn tmp_test() {
    let size = timed_maccess_template_end as *const u8 as usize
        - timed_maccess_template as *const u8 as usize;
    println!("maccess function size : {}", size);
    let size = timed_clflush_template_end as *const u8 as usize
        - timed_clflush_template as *const u8 as usize;
    println!("clflush function size : {}", size);
    let mem: u8 = 42;
    let p = &mem as *const u8;
    println!("maccess {:p} : {}", p, unsafe { timed_maccess_template(p) });
    println!("clflush {:p} : {}", p, unsafe { timed_clflush_template(p) });

    let f = Function::try_new(1, 0, TIMED_CLFLUSH).unwrap();

    println!("{:p}", f.fun as *const u8);
    let r = unsafe { (f.fun)(p) };
    println!("relocate clflush {:p}, {}", (f.fun) as *const u8, r);
}
