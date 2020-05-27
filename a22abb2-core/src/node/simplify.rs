use either::Either;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;
use std::iter;

use super::{ConstKind, Node, VarOpKind};
use super::util::is_minus_one;

pub fn simplify(node: Node) -> Node {
    match node {
        Node::Const(ConstKind::Tau) => Node::Const(ConstKind::Pi) * Node::two(),
        Node::VarOp { kind, children } => {
            let children = deep_flatten_children(children, kind)
                .into_iter()
                .map(&simplify);
            let children = collapse_numbers(children, kind);

            let mut children_by_factors: HashMap<Node, Vec<Node>> = HashMap::new();

            // count duplicate children
            for child in children {
                // TODO: better algorithm
                let (child, factor) = match kind {
                    VarOpKind::Add => match child {
                        Node::VarOp {
                            kind: VarOpKind::Mul,
                            children: mut sub_children,
                        } => {
                            match sub_children.len() {
                                2 => {
                                    let mut iter = sub_children.into_iter();
                                    let a = iter.next().unwrap();
                                    let b = iter.next().unwrap();
                                    if node_factor_heuristic(&a) > node_factor_heuristic(&b) {
                                        (a, b)
                                    } else {
                                        (b, a)
                                    }
                                }

                                len if len > 2 => {
                                    // sort so that the last one is the factor
                                    sub_children.sort_by(|a, b| node_factor_heuristic(a)
                                        .partial_cmp(&node_factor_heuristic(b))
                                        .unwrap());
                                    let factor = sub_children.pop().unwrap();
                                    let remaining = Node::VarOp {
                                        kind: VarOpKind::Mul,
                                        children: sub_children,
                                    };
                                    // TODO: better heuristics to put more
                                    //  interesting factor first
                                    (remaining, factor)
                                }

                                // There has to be at least two factors
                                // because otherwise, it would have been
                                // reduced to just a number, not a multiplication.
                                _ => panic!("multiplication with less than 2 factors"),
                            }
                        }

                        // Fallback to a factor of 1 because it doesn't
                        // change the end value.
                        child => (child, Node::one()),
                    },
                    VarOpKind::Mul => match child {
                        Node::Exp(a, b) => (*a, *b),
                        // Fallback to a power of 1 because it doesn't
                        // change the end value.
                        child => (child, Node::one()),
                    },
                };

                children_by_factors
                    .entry(child)
                    .or_insert_with(Vec::new)
                    .push(factor);
            }

            let compressed_children = children_by_factors
                .into_iter()
                .filter_map(|(child, factors)| {
                    let factors = collapse_numbers(factors.into_iter(), VarOpKind::Add);

                    // if there is only one factor, return it instead of a list to add
                    match factors.len() {
                        0 => None,
                        1 => Some(match factors.into_iter().next().unwrap() {
                            // if the only factor is 1, then return the child directly
                            Node::Num { ref val, .. } if val.is_one() => child,
                            // If the only factor is 0, then discard because 0
                            // times anything is 0.
                            Node::Num { ref val, .. } if val.is_zero() => return None,

                            other => kind.compress(child, other),
                        }),
                        _ => Some(kind.compress(
                            child,
                            Node::VarOp {
                                kind: VarOpKind::Add,
                                children: factors,
                            },
                        )),
                    }
                })
                .collect::<Vec<_>>();

            // if there is only one node, return it instead of a list to evaluate
            match compressed_children.len() {
                0 => Node::Num {
                    val: kind.identity_bigr(),
                    input_base: None,
                },
                1 => compressed_children.into_iter().next().unwrap(),
                _ => Node::VarOp {
                    kind,
                    children: compressed_children,
                },
            }
        }

        Node::Exp(a, b) => match (simplify(*a), simplify(*b)) {
            // 1^k equals 1
            (Node::Num { ref val, .. }, _) if val.is_one() => Node::one(),
            // k^0 equals 1
            (_, Node::Num { ref val, .. }) if val.is_zero() => Node::one(),
            // (c^d)^b = c^(d*b)
            (Node::Exp(c, d), b) => {
                let new_exp = simplify((*d) * b);
                if let Node::Num { val, input_base } = &new_exp {
                    // We cannot simplify if it changes the display base of the
                    // result!
                    let will_not_change_base = input_base.map(|x| x == 10)
                        .unwrap_or(true);
                    if will_not_change_base && val.is_one() {
                        // 1^k equals 1
                        return *c;
                    }
                }
                Node::Exp(c, Box::new(new_exp))
            },
            (Node::Num { val, input_base }, rhs) if is_minus_one(&rhs) => {
                let (numer, denom) = val.into();
                Node::Num {
                    // take the inverse by swapping numerator and denominator
                    val: BigRational::new(denom, numer),
                    input_base,
                }
            }
            // we cannot simplify
            (a, b) => Node::Exp(Box::new(a), Box::new(b)),
        },

        Node::Sin(ref inner) | Node::Cos(ref inner) | Node::Tan(ref inner) => {
            let inner_simplified = simplify(*inner.clone());
            if let Some(mut pi_factor) = get_pi_factor(&inner_simplified) {
                // simplify (2a + b)pi as b*pi with -1 <= b <= 1
                pi_factor %= BigRational::from_integer(2.into());
                // Map negative b's to positive, but keep the same result in
                // the end.
                if pi_factor.is_negative() {
                    pi_factor += BigRational::from_integer(2.into());
                }
                if pi_factor.is_zero() {
                    return match &node {
                        Node::Sin(_) => Node::zero(),
                        Node::Cos(_) => Node::one(),
                        Node::Tan(_) => Node::zero(),
                        _ => unreachable!(),
                    };
                } else if pi_factor.is_one() {
                    return match &node {
                        Node::Sin(_) => Node::zero(),
                        Node::Cos(_) => Node::minus_one(),
                        Node::Tan(_) => Node::zero(),
                        _ => unreachable!(),
                    };
                } else if *pi_factor.denom() == 2.into() {
                    // could be 1/2 or 3/2
                    return if pi_factor.numer().is_one() {
                        match &node {
                            Node::Sin(_) => Node::one(),
                            Node::Cos(_) => Node::zero(),
                            Node::Tan(_) => todo!("handle errors gracefully"),
                            _ => unreachable!(),
                        }
                    } else {
                        match &node {
                            Node::Sin(_) => Node::minus_one(),
                            Node::Cos(_) => Node::zero(),
                            Node::Tan(_) => todo!("handle errors gracefully"),
                            _ => unreachable!(),
                        }
                    };
                } else if *pi_factor.denom() == 3.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 1.into() && *pi_factor.numer() < 5.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 3.into();

                    return match &node {
                        Node::Sin(_) if is_top => Node::three().sqrt() / Node::two(),
                        Node::Sin(_) if !is_top => -Node::three().sqrt() / Node::two(),
                        Node::Cos(_) if !is_left => Node::two().inverse(),
                        Node::Cos(_) if is_left => -Node::two().inverse(),
                        Node::Tan(_) if is_top != is_left => Node::three().sqrt(),
                        Node::Tan(_) if is_top == is_left => -Node::three().sqrt(),
                        _ => unreachable!(),
                    };
                } else if *pi_factor.denom() == 4.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 2.into() && *pi_factor.numer() < 6.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 4.into();

                    return match &node {
                        Node::Sin(_) if is_top => Node::two().sqrt().inverse(),
                        Node::Sin(_) if !is_top => -Node::two().sqrt().inverse(),
                        Node::Cos(_) if !is_left => Node::two().sqrt().inverse(),
                        Node::Cos(_) if is_left => -Node::two().sqrt().inverse(),
                        Node::Tan(_) if is_top != is_left => Node::one(),
                        Node::Tan(_) if is_top == is_left => Node::minus_one(),
                        _ => unreachable!(),
                    };
                } else if *pi_factor.denom() == 6.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 3.into() && *pi_factor.numer() < 9.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 6.into();

                    return match &node {
                        Node::Sin(_) if is_top => Node::two().inverse(),
                        Node::Sin(_) if !is_top => -Node::two().inverse(),
                        Node::Cos(_) if !is_left => Node::three().sqrt() / Node::two(),
                        Node::Cos(_) if is_left => -Node::three().sqrt() / Node::two(),
                        Node::Tan(_) if is_top != is_left => Node::three().sqrt().inverse(),
                        Node::Tan(_) if is_top == is_left => -Node::three().sqrt().inverse(),
                        _ => unreachable!(),
                    };
                }
            }
            // failed to simplify with common angle
            match &node {
                Node::Sin(_) => Node::Sin(Box::new(inner_simplified)),
                Node::Cos(_) => Node::Cos(Box::new(inner_simplified)),
                Node::Tan(_) => Node::Tan(Box::new(inner_simplified)),
                _ => unreachable!(),
            }
        }

        // fallback to doing nothing
        _ => node,
    }
}

fn get_pi_factor(node: &Node) -> Option<BigRational> {
    match node {
        Node::Const(ConstKind::Pi) => Some(BigRational::from_integer(1.into())),
        Node::Const(ConstKind::Tau) => Some(BigRational::from_integer(2.into())),
        Node::Num { val, .. } if val.is_zero() => Some(Zero::zero()),
        Node::VarOp {
            children,
            kind: VarOpKind::Mul,
        } => {
            let mut total_factor: BigRational = One::one();
            let mut has_pi = false;
            for child in children {
                if let Node::Num { val, .. } = child {
                    if val.is_zero() {
                        // zero times anything is zero
                        return Some(Zero::zero());
                    }
                    total_factor *= val;
                } else if let Some(factor) = get_pi_factor(&child) {
                    if factor.is_zero() {
                        // zero times anything is zero
                        return Some(Zero::zero());
                    }
                    if has_pi {
                        // We already have pi, so this will be pi^2 which we
                        // do not support here because we're supposed to
                        // return the multiplier of pi as an integer.
                        return None;
                    }
                    total_factor *= factor;
                    has_pi = true;
                } else {
                    // complex node that we do not understand
                    return None;
                }
            }
            if has_pi {
                Some(total_factor)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn collapse_numbers<I>(nodes: I, kind: VarOpKind) -> Vec<Node>
where
    I: Iterator<Item = Node>,
{
    let mut result = Vec::new();
    let mut number = None;
    let mut base = None;

    for node in nodes {
        match node {
            Node::Num { val, input_base } => {
                let left = match number {
                    Some(val) => val,
                    None => kind.identity_bigr(),
                };

                number = Some(kind.eval_bigr_fn()(left, val));
                base = Node::get_op_result_base(base, input_base);
            }

            other => result.push(other),
        }
    }

    // put the result number with all of the other nodes
    if let Some(number) = number {
        result.push(Node::Num {
            val: number,
            input_base: base,
        });
    }

    result
}

/// Turns add(add(1, add(2)), 3) into add(1, 2, 3).
fn deep_flatten_children(children: Vec<Node>, op_kind: VarOpKind) -> Vec<Node> {
    let mut result = Vec::new();
    let mut remaining = children;

    while !remaining.is_empty() {
        remaining = remaining
            .into_iter()
            .flat_map(|child| {
                // a workaround to make the borrow checker happy
                let can_be_flattened = if let Node::VarOp { kind: sub_kind, .. } = &child {
                    *sub_kind == op_kind
                } else {
                    false
                };

                if can_be_flattened {
                    let sub_children = match child {
                        Node::VarOp { children: val, .. } => val,
                        _ => unreachable!(),
                    };

                    // The child can be flattened, so we will continue
                    // in the next round.
                    Either::Left(sub_children.into_iter())
                } else {
                    // the child cannot be flattened
                    result.push(child);
                    Either::Right(iter::empty())
                }
            })
            .collect::<Vec<_>>();
    }

    result
}

fn node_factor_heuristic(node: &Node) -> u32 {
    // greater numbers mean "use me as a factor" when factoring
    match node {
        Node::Sin(_) | Node::Cos(_) | Node::Tan(_) => 4,
        Node::VarOp { .. } => 3,
        Node::Const(_) => 2,
        Node::Exp(_, _) => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_simplifies_trigonometric_functions_with_common_angles() {
        // x = n*pi/6
        // 0 <= x < 2pi
        test_trigonometric_functions_on_range(0, 12, 6);

        // x = n*pi/4
        // 0 <= x < 2pi
        test_trigonometric_functions_on_range(0, 8, 4);

        // test outside [0; 2pi] range
        test_trigonometric_functions_on_range(30, 40, 4);
        // test negative
        test_trigonometric_functions_on_range(-100, -90, 3);
    }

    fn test_trigonometric_functions(input: &Node) {
        const TRIGO_FUNCS: [&dyn Fn(Box<Node>) -> Node; 3] = [&Node::Sin, &Node::Cos, &Node::Tan];
        for func in &TRIGO_FUNCS {
            let node = func(Box::new(input.clone()));
            let ground_truth = node.eval();
            if ground_truth.val.abs() > 99999999.0 {
                // impossible tangeant
                continue;
            }
            let simplified = simplify(node).eval();
            assert!(
                (simplified.val - ground_truth.val).abs() < 0.001,
                "input: {}, output: {}, expected: {}",
                input,
                simplified.val,
                ground_truth.val
            );
            assert_eq!(simplified.display_base, ground_truth.display_base);
        }
    }

    fn test_trigonometric_functions_on_range(from: i32, to: i32, denom: u32) {
        for n in from..=to {
            let input = Node::Const(ConstKind::Pi)
                * Node::Num {
                    val: BigRational::new(n.into(), denom.into()),
                    input_base: Some(16),
                };
            test_trigonometric_functions(&input);
        }
    }
}
