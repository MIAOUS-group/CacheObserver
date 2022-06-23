use crate::complex_addressing::CacheSlicing::{
    ComplexAddressing, NoSlice, SimpleAddressing, Unsupported,
};
use cpuid::{CPUVendor, MicroArchitecture};

use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug, Copy, Clone)]
pub struct SimpleAddressingParams {
    pub shift: u8, // How many trailing zeros
    pub bits: u8,  // How many ones
}

#[derive(Debug, Copy, Clone)]
pub enum CacheSlicing {
    Unsupported,
    ComplexAddressing(&'static [usize]),
    SimpleAddressing(SimpleAddressingParams),
    NoSlice,
}

/// Function known to be used on most powers of 2 core processors from Sandy Bridge to Skylake
const SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS: [usize; 4] = [
    0b0110_1101_0111_1101_0101_1101_0101_0001_000000,
    0b1011_1010_1101_0111_1110_1010_1010_0010_000000,
    0b1111_0011_0011_0011_0010_0100_1100_0100_000000,
    0b0, // TODO
];

/// Functions for crystall well
/// Not able to test bit 34
/// o0 = b10 b12 b14 b16 b17 b18 b20 b22 b24 b25 b26 b27 b28 b30 b32 b33
///
/// o1 = b11 b13 b15 b17 b19 b20 b21 b22 b23 b24 b26 b28 b29 b31 b33
const CRYSTAL_WELL_FUNCTIONS: [usize; 2] = [
    0b0000_1101_0111_1101_0101_1101_0101_0000_000000,
    0b0000_1010_1101_0111_1110_1010_1010_0000_000000,
];

/// function known to be used on core i9-9900
#[allow(non_upper_case_globals)]
const COFFEELAKE_R_i9_FUNCTIONS: [usize; 4] = [
    0b0000_1111_1111_1101_0101_1101_0101_0001_000000,
    0b0000_0110_1111_1011_1010_1100_0100_1000_000000,
    0b0000_1111_1110_0001_1111_1100_1011_0000_000000,
    0b0, // TODO
];
// missing functions for more than 8 cores.

// FIXME : Need to account for Family Model (and potentially stepping)
// Amongst other thing Crystal well products have a different function. (0x6_46)
// Same thing for Kaby Lake with 8 cores apparently.

pub fn cache_slicing(
    uarch: MicroArchitecture,
    physical_cores: u8,
    vendor: CPUVendor,
    family_model_display: u32,
    _stepping: u32,
) -> CacheSlicing {
    let trailing_zeros = physical_cores.trailing_zeros();
    if physical_cores != (1 << trailing_zeros) {
        return Unsupported;
    }

    match vendor {
        CPUVendor::Intel => {
            match uarch {
                MicroArchitecture::KabyLake | MicroArchitecture::Skylake => ComplexAddressing(
                    &SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros + 1) as usize)],
                ),
                MicroArchitecture::CoffeeLake => {
                    if family_model_display == 0x6_9E {
                        // TODO stepping should probably be involved here
                        ComplexAddressing(
                            &COFFEELAKE_R_i9_FUNCTIONS[0..((trailing_zeros + 1) as usize)],
                        )
                    } else {
                        ComplexAddressing(
                            &SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros + 1) as usize)],
                        )
                    }
                }
                MicroArchitecture::SandyBridge
                | MicroArchitecture::HaswellE
                | MicroArchitecture::Broadwell
                | MicroArchitecture::IvyBridge
                | MicroArchitecture::IvyBridgeE => ComplexAddressing(
                    &SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros) as usize)],
                ),
                MicroArchitecture::Haswell => {
                    if family_model_display == 0x06_46 {
                        ComplexAddressing(&CRYSTAL_WELL_FUNCTIONS[0..((trailing_zeros) as usize)])
                    } else {
                        ComplexAddressing(
                            &SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros) as usize)],
                        )
                    }
                }
                MicroArchitecture::Nehalem | MicroArchitecture::Westmere => {
                    Unsupported //SimpleAddressing(((physical_cores - 1) as usize) << 6 + 8) // Hardcoded for 4 cores FIXME !!!
                }
                _ => Unsupported,
            }
        }
        CPUVendor::AMD => Unsupported,
        _ => Unsupported,
    }
}

fn hash(addr: usize, mask: usize) -> usize {
    ((addr & mask).count_ones() & 1) as usize
}

impl CacheSlicing {
    pub fn can_hash(&self) -> bool {
        match self {
            Unsupported | NoSlice | SimpleAddressing(_) => false,
            ComplexAddressing(_) => true,
        }
    }
    pub fn hash(&self, addr: usize) -> Option<u8> {
        match self {
            SimpleAddressing(mask) => Some(((addr >> mask.shift) & ((1 << mask.bits) - 1)) as u8),
            ComplexAddressing(masks) => {
                let mut res = 0;
                for mask in *masks {
                    res <<= 1;
                    res |= hash(addr, *mask);
                }
                Some(res as u8)
            }
            _ => None,
        }
    }

    // Only works for Complex Addressing rn
    // May work in the future for simple.
    fn pivot(&self, mask: isize) -> Vec<(u8, isize)> {
        match self {
            ComplexAddressing(_functions) => {
                let mut matrix = Vec::new();

                let mut i = 1;
                let mut hashspace = 0;
                while i != 0 {
                    if i & mask != 0 {
                        let h = self.hash(i as usize).unwrap();

                        hashspace |= h;
                        matrix.push((h, i));
                    }
                    i <<= 1;
                }

                let mut i = 0; // current line in the matrix.
                let mut bit = 1;
                while bit != 0 {
                    if bit & hashspace != 0 {
                        let mut found_pivot = false;
                        for j in i..matrix.len() {
                            if matrix[j].0 & bit != 0 {
                                found_pivot = true;
                                if j != i {
                                    let mi = matrix[i];
                                    let mj = matrix[j];
                                    matrix[i] = mj;
                                    matrix[j] = mi;
                                }
                                break;
                            }
                        }
                        if found_pivot {
                            for j in 0..matrix.len() {
                                if j != i && bit & matrix[j].0 != 0 {
                                    matrix[j].0 ^= matrix[i].0;
                                    matrix[j].1 ^= matrix[i].1;
                                }
                            }
                            i += 1;
                        }
                    }
                    bit <<= 1;
                }
                while i < matrix.len() {
                    if matrix[i].0 != 0 {
                        panic!("Something went wrong with the pivot algorithm")
                    }
                    i += 1;
                }

                matrix
            }
            _ => panic!("Should not be called"),
        }
    }

    pub fn image(&self, mask: usize) -> Option<HashSet<u8>> {
        match self {
            ComplexAddressing(_functions) => {
                let matrix = self.pivot(mask as isize);

                let mut result = HashSet::<u8>::new();
                result.insert(0);

                for (u, _) in matrix {
                    let mut tmp = HashSet::new();
                    for v in &result {
                        tmp.insert(v ^ u);
                    }
                    result.extend(tmp);
                }
                Some(result)
            }
            _ => None,
        }
    }

    // This gives a basis of the kernel complement (n elements)

    pub fn kernel_compl_basis(&self, mask: usize) -> Option<HashMap<u8, isize>> {
        match self {
            ComplexAddressing(_functions) => {
                let matrix = self.pivot(mask as isize);
                let mut result = HashMap::new();
                for (slice, addr) in matrix {
                    if slice != 0 {
                        result.insert(slice, addr);
                    }
                }

                Some(result)
            }
            _ => None,
        }
    }

    // This gives a set that covers all possible values of the image. (All combination of basis elements on {0,1})
    // 2^n elements
    pub fn image_antecedent(&self, mask: usize) -> Option<HashMap<u8, isize>> {
        match self {
            ComplexAddressing(_functions) => {
                let matrix = self.pivot(mask as isize);

                let mut result = HashMap::<u8, isize>::new();
                result.insert(0, 0);

                for (slice_u, addr_u) in matrix {
                    if slice_u != 0 {
                        let mut tmp = HashMap::new();
                        for (slice_v, addr_v) in &result {
                            tmp.insert(slice_v ^ slice_u, addr_v ^ addr_u);
                        }
                        result.extend(tmp);
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }
}

/**
Type used to handle unsupported hash functions by using the Cache line addr as the Hash.
*/
#[derive(Debug, Copy, Clone)]
pub enum CacheAttackSlicing {
    Unsupported(usize),
    ComplexAddressing(&'static [usize]),
    SimpleAddressing(SimpleAddressingParams),
    NoSlice,
}

// TODO
impl CacheAttackSlicing {
    pub fn from(cs: CacheSlicing, cache_line_length: usize) -> CacheAttackSlicing {
        match cs {
            Unsupported => CacheAttackSlicing::Unsupported(!dbg!(dbg!(cache_line_length) - 1)),
            ComplexAddressing(ca) => CacheAttackSlicing::ComplexAddressing(ca),
            SimpleAddressing(sa) => CacheAttackSlicing::SimpleAddressing(sa),
            NoSlice => CacheAttackSlicing::NoSlice,
        }
    }

    pub fn hash(&self, addr: usize) -> usize {
        match self {
            CacheAttackSlicing::Unsupported(mask) => addr & mask,
            CacheAttackSlicing::SimpleAddressing(mask) => {
                (addr >> mask.shift) & ((1 << mask.bits) - 1)
            }
            CacheAttackSlicing::ComplexAddressing(masks) => {
                let mut res = 0;
                for mask in *masks {
                    res <<= 1;
                    res |= hash(addr, *mask);
                }
                res
            }
            CacheAttackSlicing::NoSlice => 0usize,
        }
    }

    // Only works for Complex Addressing rn
    // May work in the future for simple.
    fn pivot(&self, mask: isize) -> Vec<(usize, isize)> {
        match self {
            CacheAttackSlicing::ComplexAddressing(_)
            | CacheAttackSlicing::SimpleAddressing(_)
            | CacheAttackSlicing::Unsupported(_) => {
                let mut matrix = Vec::new();

                let mut i = 1;
                let mut hashspace = 0;
                while i != 0 {
                    if i & mask != 0 {
                        let h = self.hash(i as usize);

                        hashspace |= h;
                        matrix.push((h, i));
                    }
                    i <<= 1;
                }

                let mut i = 0; // current line in the matrix.
                let mut bit = 1;
                while bit != 0 {
                    if bit & hashspace != 0 {
                        let mut found_pivot = false;
                        for j in i..matrix.len() {
                            if matrix[j].0 & bit != 0 {
                                found_pivot = true;
                                if j != i {
                                    let mi = matrix[i];
                                    let mj = matrix[j];
                                    matrix[i] = mj;
                                    matrix[j] = mi;
                                }
                                break;
                            }
                        }
                        if found_pivot {
                            for j in 0..matrix.len() {
                                if j != i && bit & matrix[j].0 != 0 {
                                    matrix[j].0 ^= matrix[i].0;
                                    matrix[j].1 ^= matrix[i].1;
                                }
                            }
                            i += 1;
                        }
                    }
                    bit <<= 1;
                }
                while i < matrix.len() {
                    if matrix[i].0 != 0 {
                        panic!("Something went wrong with the pivot algorithm")
                    }
                    i += 1;
                }

                matrix
            }
            _ => panic!("Should not be called"),
        }
    }

    pub fn image(&self, mask: usize) -> HashSet<usize> {
        match self {
            CacheAttackSlicing::ComplexAddressing(_)
            | CacheAttackSlicing::SimpleAddressing(_)
            | CacheAttackSlicing::Unsupported(_) => {
                let matrix = self.pivot(mask as isize);

                let mut result = HashSet::<usize>::new();
                result.insert(0);

                for (u, _) in matrix {
                    let mut tmp = HashSet::new();
                    for v in &result {
                        tmp.insert(v ^ u);
                    }
                    result.extend(tmp);
                }
                result
            }
            _ => {
                let mut r = HashSet::new();
                r.insert(0);
                r
            }
        }
    }

    // This gives a basis of the kernel complement (n elements)

    pub fn kernel_compl_basis(&self, mask: usize) -> HashMap<usize, isize> {
        let mut result = HashMap::new();
        match self {
            CacheAttackSlicing::ComplexAddressing(_)
            | CacheAttackSlicing::SimpleAddressing(_)
            | CacheAttackSlicing::Unsupported(_) => {
                let matrix = self.pivot(mask as isize);

                for (slice, addr) in matrix {
                    if slice != 0 {
                        result.insert(slice, addr);
                    }
                }
            }
            _ => {
                result.insert(0, 0);
            }
        }
        result
    }

    // This gives a set that covers all possible values of the image. (All combination of basis elements on {0,1})
    // 2^n elements
    pub fn image_antecedent(&self, mask: usize) -> HashMap<usize, isize> {
        let mut result = HashMap::<usize, isize>::new();

        match self {
            CacheAttackSlicing::ComplexAddressing(_)
            | CacheAttackSlicing::SimpleAddressing(_)
            | CacheAttackSlicing::Unsupported(_) => {
                let matrix = self.pivot(mask as isize);

                result.insert(0, 0);

                for (slice_u, addr_u) in matrix {
                    if slice_u != 0 {
                        let mut tmp = HashMap::new();
                        for (slice_v, addr_v) in &result {
                            tmp.insert(slice_v ^ slice_u, addr_v ^ addr_u);
                        }
                        result.extend(tmp);
                    }
                }
            }
            _ => {
                result.insert(0, 0);
            }
        }
        result
    }
}
