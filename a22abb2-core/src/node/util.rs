use num_traits::One;

use super::Node;

pub(crate) fn is_minus_one(node: &Node) -> bool {
    if let Node::Num { val, input_base } = node {
        return ((val.denom().is_one() && *val.numer() == (-1).into())
            || (*val.denom() == (-1).into() && val.numer().is_one()))
            && input_base.is_none();
    }
    false
}
