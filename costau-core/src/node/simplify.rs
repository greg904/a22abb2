use itertools::Itertools;
use num_rational::BigRational;
use num_traits::{One, Pow, Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::convert::{TryFrom, TryInto};
use std::hash::{Hash, Hasher};
use std::iter;
use std::ops::{Add, Mul};

use super::util::{common, get_op_result_base, ratio_to_i32};
use super::{ConstKind, Node};

/// A description of an error that happened while trying to simplify a node.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimplifyError {
    ZeroToPowerOfNonPositive,
    ComplexRoot,
    Tan90Or270,
}

/// The value returned by the `simplify` function when it succeeds.
#[derive(Debug, PartialEq, Clone)]
pub struct SimplifySuccess {
    /// The simplified node.
    pub result: Node,
    /// Whether the algorithm actually did anything. If this is `false`, then
    /// the algorithm was not able to do any simplification.
    pub did_something: bool,
}

/// Simplifies the node.
/// If the algorithm fails to simplify the node because the node is already
/// simplified or because it is too complex, then the function will return a
/// success, but with `did_something` set to `false` in the success struct.
/// If the algorithm fails to simplify the node because the expression is
/// invalid, then the function will return an error.
/// Otherwise, it returns a success with the simplified node.
pub fn simplify(node: Node) -> Result<SimplifySuccess, SimplifyError> {
    match node {
        Node::Const(ConstKind::Tau) => {
            return Ok(SimplifySuccess {
                result: Node::Const(ConstKind::Pi) * common::two(),
                // We only do this translation for simplification purposes, but
                // we did not actually do anything smart here.
                did_something: false,
            });
        },
        Node::Sum(children) => return simplify_vararg_op(children, true),
        Node::Product(children) => return simplify_vararg_op(children, false),
        Node::Exp(lhs, rhs) => return simplify_exp(*lhs, *rhs),
        Node::Sin(ref inner) | Node::Cos(ref inner) | Node::Tan(ref inner) => {
            let inner_simplified = simplify(*inner.clone())?;
            if let Some(mut pi_factor) = get_pi_factor(&inner_simplified.result) {
                // simplify (2a + b)pi as b*pi with -1 <= b <= 1
                pi_factor %= BigRational::from_integer(2.into());
                // Map negative b's to positive, but keep the same result in
                // the end.
                if pi_factor.is_negative() {
                    pi_factor += BigRational::from_integer(2.into());
                }
                if pi_factor.is_zero() {
                    return Ok(SimplifySuccess {
                        result: match &node {
                            Node::Sin(_) => common::zero(),
                            Node::Cos(_) => common::one(),
                            Node::Tan(_) => common::zero(),
                            _ => unreachable!(),
                        },
                        did_something: true,
                    });
                } else if pi_factor.is_one() {
                    return Ok(SimplifySuccess {
                        result: match &node {
                            Node::Sin(_) => common::zero(),
                            Node::Cos(_) => common::minus_one(),
                            Node::Tan(_) => common::zero(),
                            _ => unreachable!(),
                        },
                        did_something: true,
                    });
                } else if *pi_factor.denom() == 2.into() {
                    // could be 1/2 or 3/2
                    let simplified_node = if pi_factor.numer().is_one() {
                        match &node {
                            Node::Sin(_) => common::one(),
                            Node::Cos(_) => common::zero(),
                            Node::Tan(_) => return Err(SimplifyError::Tan90Or270),
                            _ => unreachable!(),
                        }
                    } else {
                        match &node {
                            Node::Sin(_) => common::minus_one(),
                            Node::Cos(_) => common::zero(),
                            Node::Tan(_) => return Err(SimplifyError::Tan90Or270),
                            _ => unreachable!(),
                        }
                    };
                    return Ok(SimplifySuccess {
                        result: simplified_node,
                        did_something: true,
                    })
                } else if *pi_factor.denom() == 3.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 1.into() && *pi_factor.numer() < 5.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 3.into();

                    return Ok(SimplifySuccess {
                        result: match &node {
                            Node::Sin(_) if is_top => common::three().sqrt() / common::two(),
                            Node::Sin(_) if !is_top => -common::three().sqrt() / common::two(),
                            Node::Cos(_) if !is_left => common::two().inverse(),
                            Node::Cos(_) if is_left => -common::two().inverse(),
                            Node::Tan(_) if is_top != is_left => common::three().sqrt(),
                            Node::Tan(_) if is_top == is_left => -common::three().sqrt(),
                            _ => unreachable!(),
                        },
                        did_something: true
                    });
                } else if *pi_factor.denom() == 4.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 2.into() && *pi_factor.numer() < 6.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 4.into();

                    return Ok(SimplifySuccess {
                        result: match &node {
                            Node::Sin(_) if is_top => common::two().sqrt().inverse(),
                            Node::Sin(_) if !is_top => -common::two().sqrt().inverse(),
                            Node::Cos(_) if !is_left => common::two().sqrt().inverse(),
                            Node::Cos(_) if is_left => -common::two().sqrt().inverse(),
                            Node::Tan(_) if is_top != is_left => common::one(),
                            Node::Tan(_) if is_top == is_left => common::minus_one(),
                            _ => unreachable!(),
                        },
                        did_something: true
                    });
                } else if *pi_factor.denom() == 6.into() {
                    // pi/2 < x < 3pi/2
                    let is_left = *pi_factor.numer() > 3.into() && *pi_factor.numer() < 9.into();
                    // 0 < x < pi
                    let is_top = *pi_factor.numer() < 6.into();

                    return Ok(SimplifySuccess {
                        result: match &node {
                            Node::Sin(_) if is_top => common::two().inverse(),
                            Node::Sin(_) if !is_top => -common::two().inverse(),
                            Node::Cos(_) if !is_left => common::three().sqrt() / common::two(),
                            Node::Cos(_) if is_left => -common::three().sqrt() / common::two(),
                            Node::Tan(_) if is_top != is_left => common::three().sqrt().inverse(),
                            Node::Tan(_) if is_top == is_left => -common::three().sqrt().inverse(),
                            _ => unreachable!(),
                        },
                        did_something: true
                    });
                }
            }
            // failed to simplify with common angle
            return Ok(SimplifySuccess {
                result: match &node {
                    Node::Sin(_) => Node::Sin(Box::new(inner_simplified.result)),
                    Node::Cos(_) => Node::Cos(Box::new(inner_simplified.result)),
                    Node::Tan(_) => Node::Tan(Box::new(inner_simplified.result)),
                    _ => unreachable!(),
                },
                did_something: inner_simplified.did_something,
            });
        }

        // fallback to doing nothing
        node => return Ok(SimplifySuccess {
            result: node,
            did_something: false,
        }),
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

fn group_and_fold_numbers<I>(nodes: I, is_sum: bool) -> (Vec<Node>, bool)
where I: Iterator<Item = Node>,
{
    let mut result = Vec::new();
    let mut acc = None;
    let mut acc_base = None;
    let mut did_something = false;

    for node in nodes {
        match node {
            Node::Num { val, input_base } => {
                let f = if is_sum { Add::add } else { Mul::mul };
                acc = Some(match acc {
                    Some(lhs) => {
                        did_something = true;
                        f(lhs, val)
                    },
                    None => val,
                });
                acc_base = get_op_result_base(acc_base, input_base);
            }
            other => result.push(other),
        }
    }
    // put the result number with all of the other nodes
    if let Some(last) = acc {
        // check if the number is the identity for the operation
        if (is_sum && !last.is_zero()) || (!is_sum && !last.is_one()) {
            result.insert(0, Node::Num {
                val: last,
                input_base: acc_base,
            });
        }
    }
    (result, did_something)
}

/// Turns add(add(1, add(2)), 3) into add(1, 2, 3).
fn deep_flatten_children<I>(children: I, parent_is_sum: bool) -> (Vec<Node>, bool)
where
    I: IntoIterator<Item = Node>,
{
    let mut result = Vec::new();
    let mut did_something = false;
    for child in children.into_iter() {
        match (parent_is_sum, child) {
            (true, Node::Sum(sub_children)) | (false, Node::Product(sub_children)) => {
                let mut tmp = deep_flatten_children(sub_children, parent_is_sum);
                result.append(&mut tmp.0);
                did_something |= tmp.1;
            },
            (_, child) => result.push(child),
        }
    }
    (result, did_something)
}

fn expand_product<'a>(factors: &'a [Node]) -> Box<dyn Iterator<Item = Node> + 'a> {
    match factors {
        [] => Box::new(iter::empty()),
        [Node::Sum(terms)] => Box::new(terms.iter().cloned()),
        [Node::Sum(head_terms), tail @ ..] => {
            let tail_terms: Vec<_> = expand_product(&tail).collect();
            Box::new(head_terms.iter().cloned()
                .cartesian_product(tail_terms)
                .map(|(a, b)| a * b))
        }
        _ => Box::new(iter::once(Node::Product(factors.to_vec()))),
    }
}

fn simplify_vararg_op<I>(children: I, is_sum: bool) -> Result<SimplifySuccess, SimplifyError>
where
    I: IntoIterator<Item = Node>,
{
    let (children, mut did_something) = deep_flatten_children(children, is_sum);

    let children: Vec<SimplifySuccess> = children
        .into_iter()
        .map(simplify)
        .collect::<Result<Vec<_>, _>>()?;
    for child in children.iter() {
        did_something |= child.did_something;
    }

    if !is_sum {
        for child in children.iter() {
            if let Node::Num { val, .. } = &child.result {
                if val.is_zero() {
                    // Zero short circuits multiplication.
                    // TODO: maybe we should not be ignoring the base of the
                    //  other terms
                    return Ok(SimplifySuccess {
                        result: common::zero(),
                        did_something: children.len() > 1,
                    });
                }
            }
        }
    }

    // transform `3*2+pi*2+4+9` into `19+pi*2`
    let tmp = group_and_fold_numbers(children.into_iter().map(|c| c.result), is_sum);
    let children = tmp.0;
    did_something |= tmp.1;

    if !is_sum {
        // expand product
        let expanded_terms: Vec<Node> = expand_product(&children).collect();
        if expanded_terms.len() > 1 {
            let new_node = simplify_vararg_op(expanded_terms.into_iter(), true)?.result;
            return Ok(SimplifySuccess {
                result: new_node,
                did_something: true,
            });
        }
    }

    // In the case of multiplication, this is children by exponent
    // The second field in the value tuple is used to preserve insertion order.
    let mut children_by_factors: HashMap<Node, (Vec<Node>, usize)> = HashMap::new();
    let mut insertion_counter = 0;
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
                            if node_factor_heuristic(&a) >= node_factor_heuristic(&b) {
                                (b, a)
                            } else {
                                (a, b)
                            }
                        }

                        len if len > 2 => {
                            // sort so that the last one is the factor
                            sub_children.sort_by(|a, b| {
                                node_factor_heuristic(a)
                                    .partial_cmp(&node_factor_heuristic(b))
                                    .unwrap()
                            });
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
            .or_insert_with(|| (Vec::new(), insertion_counter)).0
            .push(factor);

        // It doesn't matter if we increment this when we did not actually
        // insert anything new.
        // The only thing that matters is that when we insert something new, the
        // counter is larger than the counter for the previously inserted
        // element.
        insertion_counter += 1;
    }

    // Sort by insertion order because the `HashMap` disorganized our nice
    // terms/factors entered by the user.
    let mut sorted_entries = children_by_factors
        .into_iter()
        .collect::<Vec<_>>();
    sorted_entries.sort_by_key(|(_, (_, inserted))| *inserted);

    let mut children = Vec::new();
    for (child, factors) in sorted_entries {
        // We always want to use addition here to fold factors:
        // - pi*3 + pi*5 = pi*(3+5)
        // - pi^3 * pi^5 = pi^(3+5)
        let tmp = group_and_fold_numbers(factors.0.into_iter(), true);
        let factors = tmp.0;
        did_something |= tmp.1;

        match factors.len() {
            0 => {},
            1 => {
                let new_child = match factors.into_iter().next().unwrap() {
                    // if the only factor is 1, then return the child directly
                    Node::Num { ref val, .. } if val.is_one() => child,
                    // If the only factor is 0, then discard because 0
                    // times anything is 0.
                    Node::Num { ref val, .. } if val.is_zero() => continue,
                    other => fold_helper(child, other, is_sum),
                };
                children.push(new_child);
            },
            _ => {
                let tmp = simplify_vararg_op(factors, true)?;
                let factor = tmp.result;
                did_something |= tmp.did_something;

                let new_child = fold_helper(child, factor, is_sum);
                children.push(new_child);
            },
        }
    }

    let result = match children.len() {
        0 => Node::Num {
            // identity
            val: if is_sum { Zero::zero() } else { One::one() },
            input_base: None,
        },
        // if there is only one node, return it instead of a list to evaluate
        1 => children.into_iter().next().unwrap(),
        _ if is_sum => Node::Sum(children),
        _ => Node::Product(children),
    };
    Ok(SimplifySuccess {
        result,
        did_something
    })
}

fn fold_helper(x: Node, factor: Node, is_sum: bool) -> Node {
    if is_sum {
        factor * x
    } else {
        Node::Exp(Box::new(x), Box::new(factor))
    }
}

fn node_factor_heuristic(node: &Node) -> i64 {
    // greater numbers mean "use me as a factor" when factoring
    match node {
        Node::Sin(_) | Node::Cos(_) | Node::Tan(_) => 5 << 32,
        Node::Sum(_) | Node::Product(_) => 4 << 32,
        Node::Const(ConstKind::Pi) => (3 << 32) + 0,
        Node::Const(ConstKind::Tau) => (3 << 32) + 1,
        Node::Const(ConstKind::E) => (3 << 32) + 2,
        Node::UnknownConst(s) => {
            // Make sure that the same constants have the same factor heuristic
            // so that they are grouped correctly.
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            let hash = hasher.finish() as u32;
            (2i64 << 32) + hash as i64
        },
        Node::Exp(_, _) => 1 << 32,
        _ => 0 << 32,
    }
}

fn simplify_exp(lhs: Node, rhs: Node) -> Result<SimplifySuccess, SimplifyError> {
    let tmp = simplify(rhs)?;
    let rhs = tmp.result;
    let mut did_something = tmp.did_something;

    // This must be done before we expand the exponent below by
    // simplifying the LHS.
    if let Node::Exp(lhs_base, lhs_exp) = lhs {
        // (a^b)^c = a^(b*c)
        let new_base = simplify(*lhs_base)?;
        did_something |= new_base.did_something;
        let new_exp = simplify((*lhs_exp) * rhs)?;
        did_something |= new_exp.did_something;
        let tmp = simplify_exp(new_base.result, new_exp.result)?;
        did_something |= tmp.did_something;
        return Ok(SimplifySuccess {
            result: tmp.result,
            did_something,
        });
    }

    let tmp = simplify(lhs)?;
    let lhs = tmp.result;
    did_something |= tmp.did_something;

    if let Node::Num { val: lhs_val, input_base: lhs_input_base } = &lhs {
        if let Node::Num { val: rhs_val, input_base: rhs_input_base } = &rhs {
            // actually try compute the exponent's result
            match simplify_exp_nums(lhs_val, rhs_val, *lhs_input_base, *rhs_input_base) {
                Some(Ok(simplified_node)) => return Ok(SimplifySuccess {
                    result: simplified_node,
                    did_something: true,
                }),
                Some(Err(err)) => return Err(err),
                None => {},
            }
        }
        if lhs_val.is_one() {
            // 1^x = 1
            return Ok(SimplifySuccess {
                result: common::one(),
                did_something: true,
            });
        }
    } else if let Node::Num { val: rhs_val, .. } = &rhs {
        if rhs_val.is_one() {
            // x^1 = x
            return Ok(SimplifySuccess {
                result: lhs,
                did_something: true,
            });
        } else if ratio_to_i32(&rhs_val) == Some(-1) {
            // (a/b)^-1 = b/a
            if let Node::Num { val: lhs_val, input_base: lhs_input_base } = &lhs {
                let inverse = Node::Num {
                    val: BigRational::new(
                        lhs_val.denom().clone(),
                        lhs_val.numer().clone(),
                    ),
                    input_base: *lhs_input_base,
                };
                return Ok(SimplifySuccess {
                    result: inverse,
                    did_something: true,
                })
            }
        } else if rhs_val.is_zero() {
            if let Node::Const(_) = &lhs {
                // our constants are never zero, so we won't have 0^0
                return Ok(SimplifySuccess {
                    result: common::one(),
                    did_something: true,
                });
            }
        }
    }

    if let Node::Num { val: rhs_val, .. } = &rhs {
        if let Some(rhs_i32) = ratio_to_i32(&rhs_val) {
            if rhs_i32 >= 2 && rhs_i32 <= 3 {
                // turn power into product of same expression multiple times
                let factors: Vec<Node> = iter::repeat(lhs)
                    .take(rhs_i32.try_into().unwrap())
                    .collect();
                return simplify_vararg_op(factors, false);
            }
        }
    }

    // failed to simplify
    Ok(SimplifySuccess {
        result: Node::Exp(Box::new(lhs), Box::new(rhs)),
        did_something,
    })
}

fn simplify_exp_nums(lhs: &BigRational, rhs: &BigRational, lhs_base: Option<u32>, rhs_base: Option<u32>) -> Option<Result<Node, SimplifyError>> {
    if lhs.is_zero() {
        if rhs.is_positive() {
            // 0^x = 0 when x > 0
            return Some(Ok(common::zero()));
        } else {
            // 0^(-1) is undefined
            return Some(Err(SimplifyError::ZeroToPowerOfNonPositive));
        }
    }
    // x^0 = 1
    if rhs.is_zero() {
        return Some(Ok(common::one()));
    }
    fn is_pow_safe(lhs_bits: usize, expon: i32) -> bool {
        // heuristic to prevent extremely big numbers
        u32::try_from(lhs_bits)
            .ok()
            .and_then(|x| 2048i32.checked_shr(x))
            .map(|x| expon.abs() <= x)
            .unwrap_or(false)
    }
    if let Some(int_expon) = ratio_to_i32(&rhs) {
        // Maybe I don't know how to use the API but I couldn't
        // manage to use the `.pow` method on `BigRational`.
        fn my_pow(a: &BigRational, expon: i32) -> BigRational {
            // this is handled in another match arm
            assert_ne!(expon, 0);
            if expon > 0 {
                BigRational::new(
                    a.numer().pow(expon as u32),
                    a.denom().pow(expon as u32),
                )
            } else {
                let pos_expon = expon.abs();
                BigRational::new(
                    a.denom().pow(pos_expon as u32),
                    a.numer().pow(pos_expon as u32),
                )
            }
        }
        let lhs_bits = lhs.denom().bits() + lhs.numer().bits();
        if is_pow_safe(lhs_bits, int_expon) {
            return Some(Ok(Node::Num {
                val: my_pow(&lhs, int_expon),
                input_base: get_op_result_base(lhs_base, rhs_base),
            }));
        }
    } else {
        let lhs = if lhs.denom().is_one() {
            Some(lhs.numer().clone())
        } else if *lhs.denom() == (-1).into() {
            Some(-lhs.numer())
        } else {
            None
        };
        let rhs_inv = if rhs.numer().is_one() {
            Some(rhs.denom().clone())
        } else if *rhs.numer() == (-1).into() {
            Some(-rhs.denom())
        } else {
            None
        };
        if let Some(lhs) = lhs {
            if let Some(rhs_inv) = rhs_inv.as_ref().and_then(ToPrimitive::to_i32) {
                // To compute x^(-1/n), we will compute x^(1/n) and
                // take the inverse of it.
                let root = rhs_inv.abs();
                let root_u32 = root.try_into().unwrap();

                if root_u32 % 2 == 0 && lhs.is_negative() {
                    return Some(Err(SimplifyError::ComplexRoot));
                }

                // Check if doing and undoing the root changes
                // the output. If it's the case, then it's
                // because we're limited by precision and we
                // won't simplify.
                let result = lhs.nth_root(root_u32);
                if is_pow_safe(result.bits(), root) {
                    let result_undo = result.pow(root_u32);
                    if result_undo == lhs {
                        // see comment above about x^(-1/n)
                        let result = if rhs_inv.is_negative() {
                            BigRational::new(One::one(), result.into())
                        } else {
                            BigRational::from_integer(result.into())
                        };
                        return Some(Ok(Node::Num {
                            val: result,
                            input_base: get_op_result_base(lhs_base, rhs_base),
                        }));
                    }
                }
            }
        }
    }

    // failed to simplify
    None
}

#[cfg(test)]
mod tests {
    use float_cmp::{ApproxEq, F64Margin};

    use super::*;
    use crate::node::EvalError;

    #[test]
    fn it_errors_with_0_to_power_of_non_positive() {
        // 2^0
        let input = Node::Exp(Box::new(common::two()), Box::new(common::zero()));
        let result = simplify(input);
        assert!(result.is_ok());

        // 0^0
        let input = Node::Exp(Box::new(common::zero()), Box::new(common::zero()));
        let result = simplify(input);
        assert_eq!(result, Err(SimplifyError::ZeroToPowerOfNonPositive));

        // 0^-2
        let input = Node::Exp(Box::new(common::zero()), Box::new(-common::two()));
        let result = simplify(input);
        assert_eq!(result, Err(SimplifyError::ZeroToPowerOfNonPositive));
    }

    #[test]
    fn it_preserves_order_when_flattening_addition() {
        // It can be annoying when you type `a + b + c + d` and it reorganizes
        // the terms during simplification. We should try to prevent this from
        // happening.
        assert_eq!(
            deep_flatten_children(
                vec![
                    Node::Const(ConstKind::Pi),
                    Node::Const(ConstKind::E) + Node::UnknownConst("hello".to_string()) + Node::Const(ConstKind::Tau),
                ],
                true
            ).0,
            vec![
                Node::Const(ConstKind::Pi),
                Node::Const(ConstKind::E),
                Node::UnknownConst("hello".to_string()),
                Node::Const(ConstKind::Tau),
            ],
        );
    }

    #[test]
    fn it_simplifies_roots() {
        let num = Node::Num {
            val: BigRational::from_integer(256.into()),
            input_base: Some(8),
        };
        // sqrt(256) = 16
        assert_eq!(
            simplify(num.clone().sqrt()).unwrap().result,
            Node::Num {
                val: BigRational::from_integer(16.into()),
                input_base: Some(8),
            }
        );
        // we cannot simplify 256^(1/3)
        assert_eq!(
            simplify(num.clone().cbrt()).unwrap().result,
            Node::Exp(
                Box::new(num.clone()),
                Box::new(simplify(common::three().inverse()).unwrap().result),
            ),
        );
        // 256^(1/4) = 4
        assert_eq!(
            simplify(Node::Exp(
                Box::new(num),
                Box::new(
                    Node::Num {
                        val: BigRational::from_integer(4.into()),
                        input_base: Some(10),
                    }
                    .inverse()
                )
            ))
            .unwrap().result,
            Node::Num {
                val: BigRational::from_integer(4.into()),
                input_base: Some(8),
            }
        );
    }

    #[test]
    fn it_expands_products() {
        let a = Node::UnknownConst("a".to_string());
        let b = Node::UnknownConst("b".to_string());
        // TODO: we can't just compare for equality for now because the terms
        //  do not have a deterministic order yet
        let ok = match simplify((a.clone() + b.clone()).sqr()) {
            Ok(SimplifySuccess { result: Node::Sum(_), .. }) => true,
            _ => false,
        };
        assert!(ok);
    }

    #[test]
    fn it_detects_impossible_real_roots() {
        // sqrt(-1) = error
        assert_eq!(
            simplify(common::minus_one().sqrt()),
            Err(SimplifyError::ComplexRoot),
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
            let ground_truth = match node.eval() {
                Ok(x) => x,
                Err(EvalError::Tan90Or270) => continue,
                Err(err) => panic!("got unexpected error: {:?}", err),
            };
            let simplified = simplify(node).unwrap().result.eval().unwrap();
            assert!(
                simplified.val.approx_eq(
                    ground_truth.val,
                    // I think that the crate is broken and doesn't calculate
                    // ULPS correctly so I've had to use the epsilon parameter.
                    F64Margin {
                        ulps: 0,
                        epsilon: 0.000001
                    }
                ),
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
