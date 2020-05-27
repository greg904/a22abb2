use either::Either;
use num_integer::Roots;
use num_rational::BigRational;
use num_traits::{One, Pow, Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::iter;
use std::ops::{Add, Mul};

use super::{ConstKind, Node};
use super::util::{get_op_result_base, is_minus_one, common};

/// Simplifies the node.
pub fn simplify(node: Node) -> Node {
    match node {
        Node::Const(ConstKind::Tau) => Node::Const(ConstKind::Pi) * common::two(),
        Node::Sum(children) => simplify_vararg_op(children, true),
        Node::Product(children) => simplify_vararg_op(children, false),
        Node::Exp(a, b) => match (simplify(*a), simplify(*b)) {
            // 1^k equals 1
            (Node::Num { ref val, .. }, _) if val.is_one() => common::one(),
            // k^0 equals 1
            (_, Node::Num { ref val, .. }) if val.is_zero() => common::one(),
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
            // try to do it manually
            (
                Node::Num { val: val_a, input_base: input_base_a },
                Node::Num { val: val_b, input_base: input_base_b },
            ) => {
                fn ratio_to_i32(ratio: &BigRational) -> Option<i32> {
                    if ratio.denom().is_one() {
                        ratio.numer().to_i32()
                    } else if *ratio.denom() == (-1).into() {
                        ratio.numer().to_i32()
                            .and_then(|x| x.checked_neg())
                    } else {
                        None
                    }
                }
                if let Some(int_expon) = ratio_to_i32(&val_b) {
                    // Maybe I don't know how to use the API but I couldn't
                    // manage to use the `.pow` method on `BigRational`.
                    fn my_pow(a: &BigRational, expon: i32) -> BigRational {
                        // this is handled in another match arm
                        assert_ne!(expon, 0);
                        if expon > 0 {
                            BigRational::new(a.numer().pow(expon as u32), a.denom().pow(expon as u32))
                        } else {
                            let pos_expon = expon.abs();
                            BigRational::new(a.denom().pow(pos_expon as u32), a.numer().pow(pos_expon as u32))
                        }
                    }
                    return Node::Num {
                        val: my_pow(&val_a, int_expon),
                        input_base: get_op_result_base(input_base_a, input_base_b),
                    };
                } else if let Some(int_base) = ratio_to_i32(&val_a) {
                    if val_b.numer().is_one() {
                        // x^(1/n) = x.nth_root(n)
                        if let Some(root) = val_b.denom().to_u32() {
                            // Check if doing and undoing the root changes the
                            // output. If it's the case, then it's because we're
                            // limited by precision and we won't simplify.
                            let result = int_base.nth_root(root);
                            let result_undo = result.checked_pow(root)
                                .expect("should not overflow because original number fitted in u32");
                            if result_undo == int_base {
                                return Node::Num {
                                    val: BigRational::from_integer(result.into()),
                                    input_base: get_op_result_base(input_base_a, input_base_b),
                                };
                            }
                        }
                    }
                }
                // cannot simplify, so repack the numbers
                Node::Exp(
                    Box::new(Node::Num { val: val_a, input_base: input_base_a }),
                    Box::new(Node::Num { val: val_b, input_base: input_base_b })
                )
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
                        Node::Sin(_) => common::zero(),
                        Node::Cos(_) => common::one(),
                        Node::Tan(_) => common::zero(),
                        _ => unreachable!(),
                    };
                } else if pi_factor.is_one() {
                    return match &node {
                        Node::Sin(_) => common::zero(),
                        Node::Cos(_) => common::minus_one(),
                        Node::Tan(_) => common::zero(),
                        _ => unreachable!(),
                    };
                } else if *pi_factor.denom() == 2.into() {
                    // could be 1/2 or 3/2
                    return if pi_factor.numer().is_one() {
                        match &node {
                            Node::Sin(_) => common::one(),
                            Node::Cos(_) => common::zero(),
                            Node::Tan(_) => todo!("handle errors gracefully"),
                            _ => unreachable!(),
                        }
                    } else {
                        match &node {
                            Node::Sin(_) => common::minus_one(),
                            Node::Cos(_) => common::zero(),
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
                        Node::Sin(_) if is_top => common::three().sqrt() / common::two(),
                        Node::Sin(_) if !is_top => -common::three().sqrt() / common::two(),
                        Node::Cos(_) if !is_left => common::two().inverse(),
                        Node::Cos(_) if is_left => -common::two().inverse(),
                        Node::Tan(_) if is_top != is_left => common::three().sqrt(),
                        Node::Tan(_) if is_top == is_left => -common::three().sqrt(),
                        _ => unreachable!(),
                    };
                } else if *pi_factor.denom() == 4.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 2.into() && *pi_factor.numer() < 6.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 4.into();

                    return match &node {
                        Node::Sin(_) if is_top => common::two().sqrt().inverse(),
                        Node::Sin(_) if !is_top => -common::two().sqrt().inverse(),
                        Node::Cos(_) if !is_left => common::two().sqrt().inverse(),
                        Node::Cos(_) if is_left => -common::two().sqrt().inverse(),
                        Node::Tan(_) if is_top != is_left => common::one(),
                        Node::Tan(_) if is_top == is_left => common::minus_one(),
                        _ => unreachable!(),
                    };
                } else if *pi_factor.denom() == 6.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 3.into() && *pi_factor.numer() < 9.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 6.into();

                    return match &node {
                        Node::Sin(_) if is_top => common::two().inverse(),
                        Node::Sin(_) if !is_top => -common::two().inverse(),
                        Node::Cos(_) if !is_left => common::three().sqrt() / common::two(),
                        Node::Cos(_) if is_left => -common::three().sqrt() / common::two(),
                        Node::Tan(_) if is_top != is_left => common::three().sqrt().inverse(),
                        Node::Tan(_) if is_top == is_left => -common::three().sqrt().inverse(),
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
        Node::Product(children) => {
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

fn group_and_fold_numbers<I, F>(nodes: I, f: F) -> Vec<Node>
where
    I: Iterator<Item = Node>,
    F: Fn(BigRational, BigRational) -> BigRational
{
    let mut result = Vec::new();
    let mut acc = None;
    let mut acc_base = None;

    for node in nodes {
        match node {
            Node::Num { val, input_base } => {
                acc = Some(match acc {
                    Some(lhs) => f(lhs, val),
                    None => val,
                });
                acc_base = get_op_result_base(acc_base, input_base);
            }
            other => result.push(other),
        }
    }
    // put the result number with all of the other nodes
    if let Some(last) = acc {
        result.push(Node::Num {
            val: last,
            input_base: acc_base,
        });
    }
    result
}

/// Turns add(add(1, add(2)), 3) into add(1, 2, 3).
fn deep_flatten_children<I>(children: I, parent_is_sum: bool) -> Vec<Node>
where I: IntoIterator<Item = Node> {
    let mut result = Vec::new();

    let mut remaining: Vec<Node> = children.into_iter().collect();
    while !remaining.is_empty() {
        remaining = remaining
            .into_iter()
            .flat_map(|child| {
                // can be flattened if it's the same type as parent
                let can_be_flattened = match (parent_is_sum, &child) {
                    (true, Node::Sum(_)) => true,
                    (false, Node::Product(_)) => true,
                    _ => false,
                };
                if can_be_flattened {
                    let sub_children = match child {
                        Node::Sum(val) => val,
                        Node::Product(val) => val,
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

fn simplify_vararg_op<I>(children: I, is_sum: bool) -> Node
where I: IntoIterator<Item = Node> {
    let children = deep_flatten_children(children, is_sum)
        .into_iter()
        .map(&simplify);
    let acc_f = if is_sum { Add::add } else { Mul::mul };
    let children = group_and_fold_numbers(children, acc_f);

    // in the case of multiplication, this is children by exponent
    let mut children_by_factors: HashMap<Node, Vec<Node>> = HashMap::new();

    for child in children {
        // TODO: better algorithm
        let (child, factor) = if is_sum {
            match child {
                Node::Product(mut sub_children) => {
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
                            let remaining = Node::Product(sub_children);
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
                child => (child, common::one()),
            }
        } else {
            match child {
                Node::Exp(a, b) => (*a, *b),
                // Fallback to a power of 1 because it doesn't
                // change the end value.
                child => (child, common::one()),
            }
        };

        children_by_factors
            .entry(child)
            .or_insert_with(Vec::new)
            .push(factor);
    }

    let children = children_by_factors
        .into_iter()
        .filter_map(|(child, factors)| {
            // We always want to use addition here to fold factors:
            // - pi*3 + pi*5 = pi*(3+5)
            // - pi^3 * pi^5 = pi^(3+5)
            let factors = group_and_fold_numbers(factors.into_iter(), Add::add);

            // if there is only one factor, return it instead of a list to add
            match factors.len() {
                0 => None,
                1 => Some(match factors.into_iter().next().unwrap() {
                    // if the only factor is 1, then return the child directly
                    Node::Num { ref val, .. } if val.is_one() => child,
                    // If the only factor is 0, then discard because 0
                    // times anything is 0.
                    Node::Num { ref val, .. } if val.is_zero() => return None,

                    other => fold_helper(child, other, is_sum),
                }),
                _ => Some(fold_helper(child, Node::Sum(factors), is_sum)),
            }
        })
        .collect::<Vec<_>>();

    // if there is only one node, return it instead of a list to evaluate
    match children.len() {
        0 => Node::Num {
            // identity
            val: if is_sum { Zero::zero() } else { One::one() },
            input_base: None,
        },
        1 => children.into_iter().next().unwrap(),
        _ if is_sum => Node::Sum(children),
        _ => Node::Product(children),
    }
}

fn fold_helper(x: Node, factor: Node, is_sum: bool) -> Node {
    if is_sum {
        x * factor
    } else {
        Node::Exp(Box::new(x), Box::new(factor))
    }
}

fn node_factor_heuristic(node: &Node) -> u32 {
    // greater numbers mean "use me as a factor" when factoring
    match node {
        Node::Sin(_) | Node::Cos(_) | Node::Tan(_) => 4,
        Node::Sum(_) | Node::Product(_) => 3,
        Node::Const(_) => 2,
        Node::Exp(_, _) => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_simplifies_roots() {
        let num = Node::Num {
            val: BigRational::from_integer(256.into()),
            input_base: Some(8),
        };
        // sqrt(256) = 16
        assert_eq!(
            simplify(num.clone().sqrt()),
            Node::Num {
                val: BigRational::from_integer(16.into()),
                input_base: Some(8),
            }
        );
        // we cannot simplify 256^(1/3)
        assert_eq!(
            simplify(num.clone().cbrt()),
            Node::Exp(
                Box::new(num.clone()),
                Box::new(simplify(common::three().inverse())),
            ),
        );
        // 256^(1/4) = 4
        assert_eq!(
            simplify(Node::Exp(
                Box::new(num),
                Box::new(Node::Num {
                    val: BigRational::from_integer(4.into()),
                    input_base: Some(10),
                }.inverse())
            )),
            Node::Num {
                val: BigRational::from_integer(4.into()),
                input_base: Some(8),
            }
        );
    }

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
            let ground_truth = node.eval().unwrap();
            if ground_truth.val.abs() > 99999999.0 {
                // impossible tangeant
                continue;
            }
            let simplified = simplify(node).eval().unwrap();
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
