//! Select a complete precomputed AdaLN plan before resolving per-token rows.
//! Equal timestep values from different GEMM batch shapes are not interchangeable.
use crate::Result;
use serde_json::Value;

#[derive(Debug, PartialEq, Eq)]
pub struct PlanSelection {
    pub plan: usize,
    pub cursors: Vec<usize>,
    pub indices: Vec<Vec<i64>>,
}

fn valid_values(values: &[f32]) -> bool {
    !values.is_empty()
        && values.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        && values.windows(2).all(|v| v[0] < v[1])
}

/// An absent/unknown plan returns `None` so callers can retain their existing
/// explicit row-mapping fallback. Malformed installed plans never fall back.
/// Matching covers every step, every value and its FP32 bits, in order.
pub fn select(adaln: &Value, request: &[(Vec<f32>, Vec<i64>)]) -> Result<Option<PlanSelection>> {
    let steps = request.len();
    if !(1..=999).contains(&steps) {
        return Err("AdaLN plan requires 1..999 model evaluations".into());
    }
    for (values, indices) in request {
        if !valid_values(values) || indices.is_empty() || indices.iter().any(|&i|
            usize::try_from(i).map_or(true, |i| i >= values.len())) {
            return Err("invalid request timestep values or row indices".into());
        }
    }
    let rows = request[0].1.len();
    if request.iter().any(|(_, indices)| indices.len() != rows) {
        return Err("request timestep row count changes between steps".into());
    }
    let entries = match adaln.get("plans") {
        None => return Ok(None),
        Some(value) => value.as_array().ok_or("AdaLN plans must be an array")?,
    };
    if adaln["num_steps"].as_u64() != Some(steps as u64) {
        return Err("installed AdaLN plan step count differs".into());
    }
    let wanted: Vec<Vec<u32>> = request.iter()
        .map(|(values, _)| values.iter().map(|v| v.to_bits()).collect()).collect();
    let mut installed: Vec<(usize, Vec<Vec<u32>>)> = Vec::new();
    let mut selected = None;
    for entry in entries {
        let plan = usize::try_from(entry["plan_index"].as_u64().ok_or("invalid AdaLN plan index")?)?;
        let base = plan.checked_mul(steps).ok_or("AdaLN plan cursor overflow")?;
        base.checked_add(steps - 1).ok_or("AdaLN plan cursor overflow")?;
        let fingerprint_steps = entry["fingerprint"].as_array().ok_or("missing AdaLN plan fingerprint")?;
        if fingerprint_steps.len() != steps {
            return Err("installed AdaLN fingerprint step count differs".into());
        }
        let mut fingerprint = Vec::new();
        for step in fingerprint_steps {
            let values = step.as_array().ok_or("invalid AdaLN step fingerprint")?.iter()
                .map(|v| v.as_f64().map(|v| v as f32).ok_or("invalid AdaLN timestep"))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if !valid_values(&values) {
                return Err("invalid installed AdaLN timestep values".into());
            }
            fingerprint.push(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        if installed.iter().any(|(index, previous)| *index == plan || *previous == fingerprint) {
            return Err("duplicate or ambiguous installed AdaLN plan".into());
        }
        if fingerprint == wanted {
            selected = Some(PlanSelection {
                plan,
                cursors: (0..steps).map(|step| base + step).collect(),
                indices: request.iter().map(|(_, indices)| indices.clone()).collect(),
            });
        }
        installed.push((plan, fingerprint));
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn public_base_and_dry_run_schedules_select_all_their_rows() {
        for steps in [1, 27, 49, 99] {
            let request=vec![(vec![0.0,0.5],vec![0,1]);steps];
            let fingerprint=vec![vec![0.0,0.5];steps];
            let manifest=json!({"num_steps":steps,"plans":[{"plan_index":1,"fingerprint":fingerprint}]});
            let selected=select(&manifest,&request).unwrap().unwrap();
            assert_eq!(selected.cursors,(steps..2*steps).collect::<Vec<_>>());
        }
    }

    fn fixture() -> (Value, Vec<(Vec<f32>, Vec<i64>)>) {
        let request: Vec<_> = (0..6).map(|step| {
            let values = if step == 0 { vec![0.0] } else { vec![0.1, 0.2] };
            let indices = if step == 0 { vec![0, 0, 0] } else { vec![0, 1, 0] };
            (values, indices)
        }).collect();
        let text: Vec<_> = request.iter().map(|(values, _)| values.clone()).collect();
        let conditioned: Vec<_> = text.iter().map(|values| {
            let mut values = values.clone(); values.push(0.999); values
        }).collect();
        (json!({"num_steps":6,"plans":[
            {"plan_index":0,"fingerprint":conditioned},
            {"plan_index":1,"fingerprint":text}]}), request)
    }

    #[test]
    fn text_only_selects_the_whole_sibling_plan() {
        let (manifest, request) = fixture();
        let plan = select(&manifest, &request).unwrap().unwrap();
        assert_eq!(plan.plan, 1);
        assert_eq!(plan.cursors, vec![6, 7, 8, 9, 10, 11]);
        assert_eq!(plan.indices, request.iter().map(|(_, rows)| rows.clone()).collect::<Vec<_>>());
    }

    #[test]
    fn conditioned_plan_and_new_sequence_lengths_keep_original_cursors() {
        let (manifest, mut request) = fixture();
        for (values, indices) in &mut request {
            values.push(0.999);
            *indices = vec![0, (values.len() - 1) as i64, 0, 1, 0];
        }
        let plan = select(&manifest, &request).unwrap().unwrap();
        assert_eq!(plan.plan, 0);
        assert_eq!(plan.cursors, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(plan.indices[0], vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn matching_is_all_steps_and_exact_float_bits() {
        let (manifest, mut request) = fixture();
        request[5].0[0] = f32::from_bits(request[5].0[0].to_bits() + 1);
        assert!(select(&manifest, &request).unwrap().is_none());
        request[5].0[0] = 0.1;
        request[0].0[0] = -0.0;
        assert!(select(&manifest, &request).unwrap().is_none());
        assert!(select(&Value::Null, &request).unwrap().is_none());
    }

    #[test]
    fn ref2va_uses_eight_steps_without_changing_six_step_admission() {
        let (mut manifest, mut request) = fixture();
        request.extend([request[5].clone(), request[5].clone()]);
        assert!(select(&Value::Null, &request).unwrap().is_none());
        assert!(select(&manifest, &request).is_err());
        manifest["num_steps"] = json!(8);
        for plan in manifest["plans"].as_array_mut().unwrap() {
            let values = plan["fingerprint"].as_array_mut().unwrap();
            values.extend([values[5].clone(), values[5].clone()]);
        }
        let selected = select(&manifest, &request).unwrap().unwrap();
        assert_eq!(selected.cursors, (8..16).collect::<Vec<_>>());
        assert_eq!(selected.indices.len(), 8);
        assert!(select(&manifest, &request[..7]).is_err());
    }

    #[test]
    fn invalid_rows_or_installed_plans_never_silently_fall_back() {
        let (manifest, request) = fixture();
        for index in [-1, 1] {
            let mut bad = request.clone(); bad[0].1[0] = index;
            assert!(select(&manifest, &bad).is_err());
        }
        let mut bad = request.clone(); bad[3].1.push(0);
        assert!(select(&manifest, &bad).is_err());
        let mut bad = request.clone(); bad[3].0[0] = f32::NAN;
        assert!(select(&manifest, &bad).is_err());
        assert!(select(&manifest, &request[..5]).is_err());
        for patch in [json!({"num_steps":5,"plans":[]}), json!({"num_steps":6,"plans":null}),
            json!({"num_steps":6,"plans":[{"plan_index":u64::MAX,"fingerprint":[]}]}),
            json!({"num_steps":6,"plans":[manifest["plans"][1].clone(),manifest["plans"][1].clone()]})] {
            assert!(select(&patch, &request).is_err());
        }
    }
}
