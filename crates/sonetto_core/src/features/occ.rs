// crates/sonetto-core/src/features/occ.rs

use crate::features::swap::pow3_u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occurrence {
    pub feature_idx: u16,
    pub pow3: u16,
}

#[derive(Clone, Debug)]
enum OccFlat {
    /// Backed by a `&'static` slice (no allocation).
    Static(&'static [Occurrence]),
    /// Heap-owned flat array (tests / tooling).
    Owned(Box<[Occurrence]>),
}

impl OccFlat {
    #[inline(always)]
    fn as_slice(&self) -> &[Occurrence] {
        match self {
            OccFlat::Static(s) => s,
            OccFlat::Owned(b) => b.as_ref(),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

/// Square -> occurrence range mapping for incremental feature updates.
///
/// This is Sonetto's equivalent of Egaroucid's `coord_to_feature` mapping:
/// for each square, it stores a list of (feature_idx, pow3(pos)) occurrences.
///
/// - `occ_off[sq]..occ_off[sq+1]` points into `occ_flat`.
/// - `pow3` is `3^pos` where `pos` is the position of the square inside the
///   *LSB-first* square list used to build the map.
#[derive(Clone, Debug)]
pub struct OccMap {
    occ_flat: OccFlat,
    occ_off: [u16; 65],
}

impl OccMap {
    /// Empty map (no features).
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            occ_flat: OccFlat::Static(&[]),
            occ_off: [0u16; 65],
        }
    }

    /// Construct an `OccMap` backed by static arrays (no allocation).
    ///
    /// Safety contract:
    /// - `occ_off` must be monotone, and `occ_off[64] == occ_flat.len()`.
    #[inline(always)]
    pub const fn from_static(occ_flat: &'static [Occurrence], occ_off: [u16; 65]) -> Self {
        Self {
            occ_flat: OccFlat::Static(occ_flat),
            occ_off,
        }
    }

    /// Build an `OccMap` from feature square-lists.
    ///
    /// `feature_squares[feature_idx][pos] = sq` (sq in 0..64).
    ///
    /// This allocates, and is intended for tests / tooling. Hot-path engines
    /// should prefer `from_static` with pre-generated mappings.
    pub fn build_from_feature_squares(feature_squares: &[&[u8]]) -> Self {
        assert!(feature_squares.len() <= (u16::MAX as usize));

        // Per-square buckets (only used during build).
        let mut buckets: Vec<Vec<Occurrence>> = (0..64).map(|_| Vec::new()).collect();

        for (feature_idx, sqs) in feature_squares.iter().enumerate() {
            let f = feature_idx as u16;
            for (pos, &sq) in sqs.iter().enumerate() {
                assert!(sq < 64, "square out of range: {sq}");
                let pow3 = pow3_u16(pos as u8);
                buckets[sq as usize].push(Occurrence { feature_idx: f, pow3 });
            }
        }

        // Build offsets.
        let mut occ_off = [0u16; 65];
        for sq in 0..64usize {
            let len = buckets[sq].len();
            let next = (occ_off[sq] as usize) + len;
            assert!(
                next <= (u16::MAX as usize),
                "OccMap too large: total occurrences exceed u16::MAX"
            );
            occ_off[sq + 1] = next as u16;
        }
        let total = occ_off[64] as usize;

        // Flatten.
        let mut occ_flat = Vec::with_capacity(total);
        for sq in 0..64usize {
            occ_flat.extend_from_slice(&buckets[sq]);
        }
        debug_assert_eq!(occ_flat.len(), total);

        Self {
            occ_flat: OccFlat::Owned(occ_flat.into_boxed_slice()),
            occ_off,
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.occ_flat.is_empty()
    }

    /// Flat backing slice of occurrences.
    ///
    /// `pub(crate)` so the hot-path incremental update code can iterate with
    /// unchecked indexing.
    #[inline(always)]
    pub(crate) fn flat(&self) -> &[Occurrence] {
        self.occ_flat.as_slice()
    }

    #[inline(always)]
    pub fn range_for_sq(&self, sq: u8) -> (usize, usize) {
        let s = self.occ_off[sq as usize] as usize;
        let e = self.occ_off[sq as usize + 1] as usize;
        (s, e)
    }

    #[inline(always)]
    pub fn occ_for_sq(&self, sq: u8) -> &[Occurrence] {
        let (s, e) = self.range_for_sq(sq);
        &self.flat()[s..e]
    }

    /// Total number of occurrences in `occ_flat`.
    #[inline(always)]
    pub fn total_occurrences(&self) -> usize {
        self.occ_off[64] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occ_map_offsets_are_monotone_and_match_flat_len() {
        let feats: Vec<&[u8]> = vec![&[0, 1, 2, 3], &[0, 8, 16, 24], &[63, 62, 61, 60, 59]];
        let occ = OccMap::build_from_feature_squares(&feats);

        assert_eq!(occ.occ_off[0], 0);
        assert_eq!(occ.occ_off[64] as usize, occ.flat().len());
        assert_eq!(occ.total_occurrences(), occ.flat().len());

        for sq in 0..64usize {
            assert!(occ.occ_off[sq] <= occ.occ_off[sq + 1]);
        }

        // sq=0 should have two occurrences (feature 0 pos0, feature 1 pos0)
        let o0 = occ.occ_for_sq(0);
        assert_eq!(o0.len(), 2);
        assert_eq!(o0[0].feature_idx, 0);
        assert_eq!(o0[0].pow3, 1);
        assert_eq!(o0[1].feature_idx, 1);
        assert_eq!(o0[1].pow3, 1);
    }

    #[test]
    fn from_static_behaves_like_owned() {
        // A tiny hand-made static map for sq0 only.
        static FLAT: [Occurrence; 2] = [
            Occurrence { feature_idx: 7, pow3: 1 },
            Occurrence { feature_idx: 9, pow3: 3 },
        ];

        const OFF: [u16; 65] = {
            let mut o = [0u16; 65];
            o[0] = 0;
            o[1] = 2;
            let mut i = 1usize;
            while i < 64 {
                o[i + 1] = 2;
                i += 1;
            }
            o
        };

        let occ = OccMap::from_static(&FLAT, OFF);

        assert_eq!(occ.total_occurrences(), 2);
        assert_eq!(occ.occ_for_sq(0).len(), 2);
        assert_eq!(occ.occ_for_sq(1).len(), 0);
    }
}
