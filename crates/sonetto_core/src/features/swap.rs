// crates/sonetto-core/src/features/swap.rs

use crate::board::Color;

/// Compute `3^len` (for `len <= 10`).
#[inline(always)]
pub const fn pow3_u32(len: u8) -> u32 {
    let mut acc = 1u32;
    let mut i = 0u8;
    while i < len {
        acc *= 3;
        i += 1;
    }
    acc
}

/// Compute `3^len` (for `len <= 10`) as `u16` (max 59049).
#[inline(always)]
pub const fn pow3_u16(len: u8) -> u16 {
    let mut acc = 1u16;
    let mut i = 0u8;
    while i < len {
        acc *= 3;
        i += 1;
    }
    acc
}

#[inline(always)]
fn swap_id_base3(mut id_abs: u32, len: u8) -> u32 {
    let mut out = 0u32;
    let mut pow = 1u32;
    let mut i = 0u8;
    while i < len {
        let digit = id_abs % 3;
        id_abs /= 3;
        let digit_swapped = match digit {
            1 => 2,
            2 => 1,
            _ => digit,
        };
        out += digit_swapped * pow;
        pow *= 3;
        i += 1;
    }
    out
}

/// Swap tables for lengths 4..=10.
///
/// `swap[len][id_abs] = id_rel_for_white` (Black is identity).
#[derive(Clone, Debug)]
pub struct SwapTables {
    swap: Vec<Vec<u32>>,
}

impl SwapTables {
    /// Build swap tables for feature lengths 4..=10.
    pub fn build_swap_tables() -> Self {
        let mut swap = vec![Vec::<u32>::new(); 11];
        for len in 4u8..=10u8 {
            let size = pow3_u32(len) as usize;
            let mut tbl = vec![0u32; size];
            for id_abs in 0..size {
                tbl[id_abs] = swap_id_base3(id_abs as u32, len);
            }
            swap[len as usize] = tbl;
        }
        Self { swap }
    }

    /// Convert an absolute ID into player-relative ID.
    ///
    /// Safety: this is a hot path, but we still guard against out-of-range
    /// `len` / `id_abs` to avoid panics if the caller has inconsistent feature
    /// metadata. In the out-of-range case we fall back to an on-the-fly swap.
    #[inline(always)]
    pub fn rel_id(&self, player: Color, len: u8, id_abs: u32) -> u32 {
        match player {
            Color::Black => id_abs,
            Color::White => {
                let li = len as usize;
                if li < self.swap.len() {
                    let tbl = &self.swap[li];
                    let ia = id_abs as usize;
                    if ia < tbl.len() {
                        return tbl[ia];
                    }
                }
                // Fallback: compute directly (still deterministic).
                swap_id_base3(id_abs, len)
            }
        }
    }

    /// Fast path for Egev2-style feature lengths (4..=10).
    ///
    /// This avoids bounds checks and is intended for extremely hot evaluation loops.
    ///
    /// # Safety
    /// - `len` must be in `4..=10`.
    /// - `id_abs` must be < `3^len`.
    #[inline(always)]
    pub unsafe fn rel_id_fast_egev2(&self, player: Color, len: u8, id_abs: u32) -> u32 {
        match player {
            Color::Black => id_abs,
            Color::White => {
                debug_assert!((4..=10).contains(&len));
                let tbl: &Vec<u32> = match len {
                    4 => self.swap.get_unchecked(4),
                    5 => self.swap.get_unchecked(5),
                    6 => self.swap.get_unchecked(6),
                    7 => self.swap.get_unchecked(7),
                    8 => self.swap.get_unchecked(8),
                    9 => self.swap.get_unchecked(9),
                    10 => self.swap.get_unchecked(10),
                    _ => core::hint::unreachable_unchecked(),
                };
                let ia = id_abs as usize;
                debug_assert!(ia < tbl.len());
                *tbl.get_unchecked(ia)
            }
        }
    }

}

/// Convenience wrapper (often matches existing call sites).
#[inline(always)]
pub fn rel_id(player: Color, len: u8, id_abs: u32, swap: &SwapTables) -> u32 {
    swap.rel_id(player, len, id_abs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_is_involution_for_supported_lengths() {
        let st = SwapTables::build_swap_tables();
        for len in 4u8..=10u8 {
            let tbl = &st.swap[len as usize];
            assert!(!tbl.is_empty());
            let size = tbl.len();
            let step = (size / 97).max(1);
            for id in (0..size).step_by(step) {
                let swapped = tbl[id] as usize;
                let swapped_back = tbl[swapped] as usize;
                assert_eq!(swapped_back, id, "len={len} id={id}");
            }
        }
    }

    #[test]
    fn rel_id_black_is_identity_white_matches_table() {
        let st = SwapTables::build_swap_tables();
        let len = 8u8;
        let size = pow3_u32(len) as usize;
        for id in (0..size).step_by(137) {
            assert_eq!(st.rel_id(Color::Black, len, id as u32), id as u32);
            assert_eq!(
                st.rel_id(Color::White, len, id as u32),
                st.swap[len as usize][id]
            );
        }
    }
}
