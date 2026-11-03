//! Opt-in sampling cap for isolated model-switch performance endpoints.
//! Retain the original recipe, weights and first evaluations. Ordinary serving
//! has no cap; reference replay must never silently become a shortened run.
pub fn steps(requested: usize, reference_replay: bool) -> Result<usize, String> {
    capped_steps(requested, reference_replay, std::env::var("NATIVE_SWITCH_BENCH_STEPS").ok().as_deref())
}

fn capped_steps(requested: usize, reference_replay: bool, cap: Option<&str>) -> Result<usize, String> {
    match cap {
        None => Ok(requested),
        Some("2") if !reference_replay && requested > 0 => Ok(requested.min(2)),
        Some("2") if reference_replay => Err("switch benchmark cap cannot be used for reference replay".into()),
        _ => Err("NATIVE_SWITCH_BENCH_STEPS must be unset or exactly 2".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_recipes_and_reference_counts_are_unchanged() {
        for count in [1, 2, 3, 6, 8, 11] {
            assert_eq!(capped_steps(count, false, None).unwrap(), count);
            assert_eq!(capped_steps(count, true, None).unwrap(), count);
        }
    }
    #[test]
    fn performance_cap_bounds_each_phase_without_extending_short_phases() {
        for count in [1, 2, 3, 6, 8, 11] {
            assert_eq!(capped_steps(count, false, Some("2")).unwrap(), count.min(2));
        }
    }
    #[test]
    fn rejects_reference_caps_and_malformed_configuration() {
        assert!(capped_steps(8, true, Some("2")).is_err());
        assert!(capped_steps(0, false, Some("2")).is_err());
        for cap in ["", "0", "1", "3", "two"] {
            assert!(capped_steps(8, false, Some(cap)).is_err());
        }
    }
}
