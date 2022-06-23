#!/bin/sh

grep '^SingleIP0:' "$1.log" | cut -b 11- > "$1.S0.csv"
grep '^UniqueIPs0:' "$1.log" | cut -b 12- > "$1.U0.csv"
grep '^SingleIP1:' "$1.log" | cut -b 11- > "$1.S1.csv"
grep '^UniqueIPs1:' "$1.log" | cut -b 12- > "$1.U1.csv"
grep '^SingleIP2:' "$1.log" | cut -b 11- > "$1.S2.csv"
grep '^UniqueIPs2:' "$1.log" | cut -b 12- > "$1.U2.csv"

grep '^SingleIPA1:'  "$1.log" | cut -b 12- > "$1.SA1.csv"
grep '^UniqueIPsA1:' "$1.log" | cut -b 13- > "$1.UA1.csv"
grep '^SingleIPA2:'  "$1.log" | cut -b 12- > "$1.SA2.csv"
grep '^UniqueIPsA2:' "$1.log" | cut -b 13- > "$1.UA2.csv"
grep '^SingleIPA3:'  "$1.log" | cut -b 12- > "$1.SA3.csv"
grep '^UniqueIPsA3:' "$1.log" | cut -b 13- > "$1.UA3.csv"
grep '^SingleIPA4:'  "$1.log" | cut -b 12- > "$1.SA4.csv"
grep '^UniqueIPsA4:' "$1.log" | cut -b 13- > "$1.UA4.csv"
grep '^SingleIPA8:'  "$1.log" | cut -b 12- > "$1.SA8.csv"
grep '^UniqueIPsA8:' "$1.log" | cut -b 13- > "$1.UA8.csv"
grep '^SingleIPB1:'  "$1.log" | cut -b 12- > "$1.SB1.csv"
grep '^UniqueIPsB1:' "$1.log" | cut -b 13- > "$1.UB1.csv"
grep '^SingleIPB2:'  "$1.log" | cut -b 12- > "$1.SB2.csv"
grep '^UniqueIPsB2:' "$1.log" | cut -b 13- > "$1.UB2.csv"
grep '^SingleIPB3:'  "$1.log" | cut -b 12- > "$1.SB3.csv"
grep '^UniqueIPsB3:' "$1.log" | cut -b 13- > "$1.UB3.csv"
grep '^SingleIPB4:'  "$1.log" | cut -b 12- > "$1.SB4.csv"
grep '^UniqueIPsB4:' "$1.log" | cut -b 13- > "$1.UB4.csv"
grep '^SingleIPB8:'  "$1.log" | cut -b 12- > "$1.SB8.csv"
grep '^UniqueIPsB8:' "$1.log" | cut -b 13- > "$1.UB8.csv"
grep '^SingleIPC1:'  "$1.log" | cut -b 12- > "$1.SC1.csv"
grep '^UniqueIPsC1:' "$1.log" | cut -b 13- > "$1.UC1.csv"
grep '^SingleIPC2:'  "$1.log" | cut -b 12- > "$1.SC2.csv"
grep '^UniqueIPsC2:' "$1.log" | cut -b 13- > "$1.UC2.csv"
grep '^SingleIPC3:'  "$1.log" | cut -b 12- > "$1.SC3.csv"
grep '^UniqueIPsC3:' "$1.log" | cut -b 13- > "$1.UC3.csv"
grep '^SingleIPC4:'  "$1.log" | cut -b 12- > "$1.SC4.csv"
grep '^UniqueIPsC4:' "$1.log" | cut -b 13- > "$1.UC4.csv"
grep '^SingleIPC8:'  "$1.log" | cut -b 12- > "$1.SC8.csv"
grep '^UniqueIPsC8:' "$1.log" | cut -b 13- > "$1.UC8.csv"
grep '^SingleIPD1:'  "$1.log" | cut -b 12- > "$1.SD1.csv"
grep '^UniqueIPsD1:' "$1.log" | cut -b 13- > "$1.UD1.csv"
grep '^SingleIPD2:'  "$1.log" | cut -b 12- > "$1.SD2.csv"
grep '^UniqueIPsD2:' "$1.log" | cut -b 13- > "$1.UD2.csv"
grep '^SingleIPD3:'  "$1.log" | cut -b 12- > "$1.SD3.csv"
grep '^UniqueIPsD3:' "$1.log" | cut -b 13- > "$1.UD3.csv"
grep '^SingleIPD4:'  "$1.log" | cut -b 12- > "$1.SD4.csv"
grep '^UniqueIPsD4:' "$1.log" | cut -b 13- > "$1.UD4.csv"
grep '^SingleIPD8:'  "$1.log" | cut -b 12- > "$1.SD8.csv"
grep '^UniqueIPsD8:' "$1.log" | cut -b 13- > "$1.UD8.csv"
grep '^SingleIPE2:'  "$1.log" | cut -b 12- > "$1.SE2.csv"
grep '^UniqueIPsE2:' "$1.log" | cut -b 13- > "$1.UE2.csv"
grep '^SingleIPE3:'  "$1.log" | cut -b 12- > "$1.SE3.csv"
grep '^UniqueIPsE3:' "$1.log" | cut -b 13- > "$1.UE3.csv"
grep '^SingleIPE4:'  "$1.log" | cut -b 12- > "$1.SE4.csv"
grep '^UniqueIPsE4:' "$1.log" | cut -b 13- > "$1.UE4.csv"

grep '^SingleIPF1:' "$1.log" | cut -b 12- > "$1.SF1.csv"
grep '^SingleIPF-1:' "$1.log" | cut -b 13- > "$1.SF-1.csv"
grep '^SingleIPF2:' "$1.log" | cut -b 12- > "$1.SF2.csv"
grep '^SingleIPF-2:' "$1.log" | cut -b 13- > "$1.SF-2.csv"
grep '^SingleIPF3:' "$1.log" | cut -b 12- > "$1.SF3.csv"
grep '^SingleIPF-3:' "$1.log" | cut -b 13- > "$1.SF-3.csv"
grep '^SingleIPF4:' "$1.log" | cut -b 12- > "$1.SF4.csv"
grep '^SingleIPF-4:' "$1.log" | cut -b 13- > "$1.SF-4.csv"
grep '^UniqueIPsF1:' "$1.log" | cut -b 13- > "$1.UF1.csv"
grep '^UniqueIPsF-1:' "$1.log" | cut -b 14- > "$1.UF-1.csv"
grep '^UniqueIPsF2:' "$1.log" | cut -b 13- > "$1.UF2.csv"
grep '^UniqueIPsF-2:' "$1.log" | cut -b 14- > "$1.UF-2.csv"
grep '^UniqueIPsF3:' "$1.log" | cut -b 13- > "$1.UF3.csv"
grep '^UniqueIPsF-3:' "$1.log" | cut -b 14- > "$1.UF-3.csv"
grep '^UniqueIPsF4:' "$1.log" | cut -b 13- > "$1.UF4.csv"
grep '^UniqueIPsF-4:' "$1.log" | cut -b 14- > "$1.UF-4.csv"


#SingleIPA1Functions:i,Addr
#UniqueIPsA1Functions:i,Addr
#SingleIPA2Functions:i,Addr
#UniqueIPsA2Functions:i,Addr
#SingleIPA3Functions:i,Addr
#UniqueIPsA3Functions:i,Addr
#SingleIPA4Functions:i,Addr
#UniqueIPsA4Functions:i,Addr
#SingleIPA8Functions:i,Addr
#UniqueIPsA8Functions:i,Addr
#SingleIPB1Functions:i,Addr
#UniqueIPsB1Functions:i,Addr
#SingleIPB2Functions:i,Addr
#UniqueIPsB2Functions:i,Addr
#SingleIPB3Functions:i,Addr
#UniqueIPsB3Functions:i,Addr
#SingleIPB4Functions:i,Addr
#UniqueIPsB4Functions:i,Addr
#SingleIPB8Functions:i,Addr
#UniqueIPsB8Functions:i,Addr
#SingleIPC1Functions:i,Addr
#UniqueIPsC1Functions:i,Addr
#SingleIPC2Functions:i,Addr
#UniqueIPsC2Functions:i,Addr
#SingleIPC3Functions:i,Addr
#UniqueIPsC3Functions:i,Addr
#SingleIPC4Functions:i,Addr
#UniqueIPsC4Functions:i,Addr
#SingleIPC8Functions:i,Addr
#UniqueIPsC8Functions:i,Addr
#SingleIPD1Functions:i,Addr
#UniqueIPsD1Functions:i,Addr
#SingleIPD2Functions:i,Addr
#UniqueIPsD2Functions:i,Addr
#SingleIPD3Functions:i,Addr
#UniqueIPsD3Functions:i,Addr
#SingleIPD4Functions:i,Addr
#UniqueIPsD4Functions:i,Addr
#SingleIPD8Functions:i,Addr
#UniqueIPsD8Functions:i,Addr
#SingleIPE2Functions:i,Addr
#UniqueIPsE2Functions:i,Addr
#SingleIPE3Functions:i,Addr
#UniqueIPsE3Functions:i,Addr
#SingleIPE4Functions:i,Addr
#UniqueIPsE4Functions:i,Addr


#grep '^SingleIP0:' "$1.log" | cut -b 11- > "$1.S0.csv"
#grep '^UniqueIPs0:' "$1.log" | cut -b 12- > "$1.U0.csv"
#grep '^SingleIP1:' "$1.log" | cut -b 11- > "$1.S1.csv"
#grep '^UniqueIPs1:' "$1.log" | cut -b 12- > "$1.U1.csv"
#grep '^SingleIP2:' "$1.log" | cut -b 11- > "$1.S2.csv"
#grep '^UniqueIPs2:' "$1.log" | cut -b 12- > "$1.U2.csv"
