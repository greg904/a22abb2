use num_traits::One;

use super::{EvalError, EvalSuccess, Node};

pub(crate) fn is_minus_one(node: &Node) -> bool {
    if let Node::Num { val, .. } = node {
        return (val.denom().is_one() && *val.numer() == (-1).into())
            || (*val.denom() == (-1).into() && val.numer().is_one());
    }
    false
}

pub(crate) fn is_two(node: &Node) -> bool {
    if let Node::Num { val, .. } = node {
        return (val.denom().is_one() && *val.numer() == 2.into())
            || (*val.denom() == (-1).into() && *val.numer() == (-2).into());
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

pub(crate) fn fold_nodes<'a, I, F>(nodes: I, init: f64, f: F) -> Result<EvalSuccess, EvalError>
where
    I: Iterator<Item = &'a Node>,
    F: Fn(f64, f64) -> f64,
{
    let mut acc = init;
    let mut acc_base = None;
    for n in nodes {
        let eval = n.eval()?;
        acc = f(acc, eval.val);
        if acc.is_nan() {
            // multiplied inf*0, probably because there was a very big number
            acc = 0.0;
        }
        acc_base = get_op_result_base(acc_base, eval.display_base);
    }
    Ok(EvalSuccess {
        val: acc,
        display_base: acc_base,
    })
}

pub(crate) mod common {
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    use crate::node::Node;

    pub(crate) fn zero() -> Node {
        Node::Num {
            val: Zero::zero(),
            input_base: None,
        }
    }

    pub(crate) fn one() -> Node {
        Node::Num {
            val: One::one(),
            input_base: None,
        }
    }

    pub(crate) fn two() -> Node {
        Node::Num {
            val: BigRational::from_integer(2.into()),
            input_base: None,
        }
    }

    pub(crate) fn three() -> Node {
        Node::Num {
            val: BigRational::from_integer(3.into()),
            input_base: None,
        }
    }

    pub(crate) fn minus_one() -> Node {
        Node::Num {
            val: -BigRational::one(),
            input_base: None,
        }
    }
}
