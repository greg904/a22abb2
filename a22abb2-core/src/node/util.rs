use num_traits::One;

use super::{EvalResult, Node};

pub(crate) fn is_minus_one(node: &Node) -> bool {
    if let Node::Num { val, input_base } = node {
        return ((val.denom().is_one() && *val.numer() == (-1).into())
            || (*val.denom() == (-1).into() && val.numer().is_one()))
            && input_base.is_none();
    }
    false
}

pub(crate) fn get_op_result_base(a_base: Option<u32>, b_base: Option<u32>) -> Option<u32> {
    match (a_base, b_base) {
        (Some(val), None) | (None, Some(val)) => Some(val),

        // prefer the more interesting bases
        (Some(10), Some(other)) | (Some(other), Some(10)) => Some(other),
        (Some(2), _) | (_, Some(2)) => Some(2),

        (Some(a), Some(_)) => Some(a), // prefer the base of the first term
        (None, None) => None,
    }
}

pub(crate) fn fold_nodes<'a, I, F>(nodes: I, init: f64, f: F) -> EvalResult
where I: Iterator<Item = &'a Node>,
      F: Fn(f64, f64) -> f64
{
    let mut acc = init;
    let mut acc_base = None;
    for n in nodes {
        let eval = n.eval();
        acc = f(acc, eval.val);
        acc_base = get_op_result_base(acc_base, eval.display_base);
    }
    EvalResult {
        val: acc,
        display_base: acc_base,
    }
}
