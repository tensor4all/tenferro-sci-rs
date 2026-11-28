//! The ring buffer of the stored curvature pairs `(sᵢ, yᵢ)` and the skip rule that guards it.
//!
//! Written for tensor4all/tenferro-sci-rs#40. The ring buffer is written independently from the storage scheme of
//! J. Nocedal, Math. Comp. 35(151), 773–782, 1980 (the `m` newest pairs, oldest overwritten first); no source code was
//! translated for it. The skip test `yᵀs ≤ eps · (−gᵀs)` and the scaling
//! `γ = yᵀs / yᵀy` of the newest pair are derived from reading SciPy v1.16.3 (commit
//! `b9105ccc2237f57acb1060202cd77f6dd264fb34`) `scipy/optimize/__lbfgsb.c` (`mainlb`, LINE777: `dr <= epsmach*ddum`;
//! `matupd`: `theta = rr/dr`); SciPy's storage is the compact representation, this is a ring buffer.

use crate::vector::Vector;

/// SciPy's `epsmach` (`__lbfgsb.c`, `np.finfo(np.float64).eps`).
pub(crate) const EPSMACH: f64 = f64::EPSILON;

/// What [`Memory::push`] did with a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Push {
    /// The pair was stored (the oldest was dropped if the buffer was full).
    Stored,
    /// `yᵀs ≤ eps · (−gᵀs)`: the pair was skipped and nothing changed.
    Skipped,
}

/// At most `m` pairs, oldest overwritten first, in slots allocated once by cloning a template vector.
#[derive(Clone, Debug)]
pub(crate) struct Memory<V: Vector> {
    s: Vec<V>,
    y: Vec<V>,
    rho: Vec<f64>,
    len: usize,
    head: usize,
    gamma: f64,
}

impl<V: Vector> Memory<V> {
    /// An empty memory of capacity `m ≥ 1`, with every slot a clone of `template`.
    pub(crate) fn new(m: usize, template: &V) -> Self {
        Self {
            s: vec![template.clone(); m],
            y: vec![template.clone(); m],
            rho: vec![0.0; m],
            len: 0,
            head: 0,
            gamma: 1.0,
        }
    }

    /// The capacity `m`.
    pub(crate) fn capacity(&self) -> usize {
        self.s.len()
    }

    /// The number of stored pairs (SciPy's `col`).
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Forgets every pair and resets `γ` to 1 (SciPy: `col = 0, head = 0, theta = 1`).
    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.head = 0;
        self.gamma = 1.0;
    }

    /// `γ = yᵀs / yᵀy` of the newest pair, 1 if there is none.
    pub(crate) fn gamma(&self) -> f64 {
        self.gamma
    }

    /// `ρ = 1 / yᵀs` of the `i`-th pair, `i = 0` the oldest.
    pub(crate) fn rho(&self, i: usize) -> Option<f64> {
        (i < self.len).then(|| self.rho[(self.head + i) % self.capacity()])
    }

    /// The `i`-th stored pair `(s, y)`, `i = 0` the oldest.
    pub(crate) fn pair(&self, i: usize) -> Option<(&V, &V)> {
        if i < self.len {
            let slot = (self.head + i) % self.capacity();
            Some((&self.s[slot], &self.y[slot]))
        } else {
            None
        }
    }

    /// Offers the pair `(s, y)` of an accepted step. `dr` is `yᵀs` as the line search measured it
    /// (`(g_new·d − g_old·d) α`), `ddum` is `−g_oldᵀs ≥ 0` and `rr` is `yᵀy`. The pair is skipped if
    /// `dr ≤ EPSMACH · ddum` (NaN `dr` compares false and is stored: the caller rejects non-finite values earlier).
    /// Otherwise it becomes the newest pair, `ρ = 1/dr`, `γ = dr/rr`.
    pub(crate) fn push(&mut self, s: &V, y: &V, dr: f64, ddum: f64, rr: f64) -> Push {
        if dr <= EPSMACH * ddum {
            return Push::Skipped;
        }
        let m = self.capacity();
        let slot = if self.len < m {
            let slot = (self.head + self.len) % m;
            self.len += 1;
            slot
        } else {
            // full: the oldest pair's slot becomes the newest
            let slot = self.head;
            self.head = (self.head + 1) % m;
            slot
        };
        if let (Some(ss), Some(ys), Some(r)) = (
            self.s.get_mut(slot),
            self.y.get_mut(slot),
            self.rho.get_mut(slot),
        ) {
            ss.clone_from(s);
            ys.clone_from(y);
            *r = 1.0 / dr;
        }
        self.gamma = dr / rr;
        Push::Stored
    }
}

#[cfg(test)]
mod tests {
    //! Validates the ring buffer (the m newest pairs, oldest first, `len ≤ m`), the skip rule
    //! `dr ≤ EPSMACH · ddum` including its boundary, `ρ = 1/dr`, `γ = dr/rr` of the newest pair and `clear`.
    //! Exact statements (decisions and inequalities), no tolerances. Source: SciPy v1.16.3 `__lbfgsb.c`
    //! `mainlb` LINE777 (`dr <= epsmach*ddum`, `nskip`), `matupd` (`theta = rr/dr`, `itail`/`head` rotation).
    use super::*;

    /// A pair `s = [k, 0]`, `y = [k, 0]` with `dr = k²`, `rr = k²`.
    fn push_k(mem: &mut Memory<Vec<f64>>, k: f64) -> Push {
        let (s, y) = (vec![k, 0.0], vec![k, 0.0]);
        mem.push(&s, &y, k * k, 0.0, k * k)
    }

    #[test]
    fn a_fresh_memory_is_empty_with_gamma_one() {
        let mem = Memory::new(3, &vec![0.0, 0.0]);
        assert_eq!((mem.capacity(), mem.len(), mem.gamma()), (3, 0, 1.0));
        assert!(mem.pair(0).is_none() && mem.rho(0).is_none());
    }

    #[test]
    fn the_ring_buffer_keeps_the_m_newest_pairs_oldest_first() {
        let mut mem = Memory::new(3, &vec![0.0, 0.0]);
        for k in 1..=7 {
            assert_eq!(push_k(&mut mem, k as f64), Push::Stored);
            assert_eq!(mem.len(), k.min(3), "after {k} pushes");
        }
        // pairs 5, 6, 7 remain, in this order
        for (i, k) in [5.0, 6.0, 7.0].into_iter().enumerate() {
            let (s, y) = mem.pair(i).expect("stored");
            assert_eq!((s, y), (&vec![k, 0.0], &vec![k, 0.0]), "slot {i}");
            assert_eq!(mem.rho(i), Some(1.0 / (k * k)), "rho {i}");
        }
        assert!(mem.pair(3).is_none());
    }

    #[test]
    fn a_memory_of_one_pair_always_holds_the_newest() {
        let mut mem = Memory::new(1, &vec![0.0, 0.0]);
        for k in 1..=4 {
            assert_eq!(push_k(&mut mem, k as f64), Push::Stored);
            assert_eq!(mem.len(), 1);
            assert_eq!(mem.pair(0).expect("stored").0, &vec![k as f64, 0.0]);
        }
    }

    #[test]
    fn gamma_is_dr_over_rr_of_the_newest_pair() {
        let mut mem = Memory::new(2, &vec![0.0]);
        assert_eq!(
            mem.push(&vec![1.0], &vec![4.0], 4.0, 0.0, 16.0),
            Push::Stored
        );
        assert_eq!(mem.gamma(), 0.25);
        assert_eq!(
            mem.push(&vec![1.0], &vec![2.0], 2.0, 0.0, 8.0),
            Push::Stored
        );
        assert_eq!(mem.gamma(), 0.25);
        assert_eq!(
            mem.push(&vec![1.0], &vec![1.0], 1.0, 0.0, 4.0),
            Push::Stored
        );
        assert_eq!(
            mem.gamma(),
            0.25,
            "γ of the newest pair only (the older ones are dropped or kept, not averaged)"
        );
        assert_eq!(
            mem.push(&vec![1.0], &vec![1.0], 3.0, 0.0, 4.0),
            Push::Stored
        );
        assert_eq!(mem.gamma(), 0.75);
    }

    #[test]
    fn the_skip_rule_is_dr_less_or_equal_eps_times_ddum() {
        let mut mem = Memory::new(2, &vec![0.0]);
        assert_eq!(
            mem.push(&vec![1.0], &vec![2.0], 2.0, 1.0, 4.0),
            Push::Stored
        );
        let (len, gamma) = (mem.len(), mem.gamma());
        // on the boundary dr == eps·ddum: skipped (≤), nothing changes
        assert_eq!(
            mem.push(&vec![9.0], &vec![9.0], EPSMACH * 1.0, 1.0, 81.0),
            Push::Skipped
        );
        assert_eq!((mem.len(), mem.gamma()), (len, gamma));
        assert_eq!(
            mem.pair(0).expect("stored").0,
            &vec![1.0],
            "the stored pair is untouched"
        );
        // one ulp above the boundary: stored
        let above = EPSMACH * (1.0 + EPSMACH);
        assert!(above > EPSMACH);
        assert_eq!(
            mem.push(&vec![9.0], &vec![9.0], above, 1.0, 81.0),
            Push::Stored
        );
        assert_eq!(mem.len(), 2);
    }

    #[test]
    fn non_positive_curvature_is_skipped() {
        let mut mem = Memory::new(2, &vec![0.0]);
        for dr in [-1.0, 0.0, -1e-300] {
            assert_eq!(
                mem.push(&vec![1.0], &vec![-1.0], dr, 0.0, 1.0),
                Push::Skipped,
                "dr = {dr}"
            );
        }
        assert_eq!(mem.len(), 0);
        // ddum = 0 and dr = 0: 0 <= 0 skips as well
        assert_eq!(
            mem.push(&vec![1.0], &vec![0.0], 0.0, 0.0, 0.0),
            Push::Skipped
        );
    }

    #[test]
    fn clear_forgets_the_pairs_and_resets_gamma() {
        let mut mem = Memory::new(2, &vec![0.0, 0.0]);
        push_k(&mut mem, 3.0);
        push_k(&mut mem, 4.0);
        push_k(&mut mem, 5.0);
        mem.clear();
        assert_eq!((mem.len(), mem.gamma()), (0, 1.0));
        push_k(&mut mem, 8.0);
        assert_eq!(mem.len(), 1);
        assert_eq!(
            mem.pair(0).expect("stored").0,
            &vec![8.0, 0.0],
            "the first pair after a reset is the oldest"
        );
    }
}
