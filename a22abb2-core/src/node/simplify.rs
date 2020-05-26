use either::Either;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use std::collections::HashMap;
use std::iter;

use super::{Node, VarOpKind, ConstKind};

pub fn simplify(node: Node) -> Node {
    match node {
        Node::VarOp { kind, children } => {
            let children = deep_flatten_children(children, kind)
                .into_iter()
                .map(&simplify);
            let children = collapse_numbers(children, kind);

            let mut children_by_factors: HashMap<Node, Vec<Node>> = HashMap::new();

            // count duplicate children
            for child in children {
                // TODO: detect nested duplicate children
                let (factor, child) = match kind {
                    VarOpKind::Add => match child {
                        Node::VarOp {
                            kind: VarOpKind::Mul,
                            children: sub_children,
                        } => {
                            match sub_children.len() {
                                2 => {
                                    let mut iter = sub_children.into_iter();
                                    (iter.next().unwrap(), iter.next().unwrap())
                                }

                                len if len > 2 => {
                                    let mut iter = sub_children.into_iter();
                                    let first = iter.next().unwrap();
                                    let remaining = Node::VarOp {
                                        kind: VarOpKind::Mul,
                                        children: iter.collect::<Vec<_>>(),
                                    };
                                    (first, remaining)
                                }

                                // There has to be at least two factors
                                // because otherwise, it would have been
                                // reduced to just a number, not a multiplication.
                                _ => panic!("multiplication with less than 2 factors"),
                            }
                        }

                        // Fallback to a factor of 1 because it doesn't
                        // change the end value.
                        child => (Node::one(), child),
                    },

                    VarOpKind::Mul => match child {
                        Node::Exp(a, b) => (*b, *a),

                        // Fallback to a power of 1 because it doesn't
                        // change the end value.
                        child => (Node::one(), child),
                    },
                };

                children_by_factors
                    .entry(child)
                    .or_insert_with(|| vec![])
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
                }).collect::<Vec<_>>();

            // if there is only one node, return it instead of a list to evaluate
            match compressed_children.len() {
                0 => Node::Num {
                    val: kind.identity_bigr(),
                    input_base: None,
                },
                1 => compressed_children.into_iter().next().unwrap(),
                _ => Node::VarOp { kind, children: compressed_children},
            }
        }

        Node::Exp(a, b) => match (simplify(*a), simplify(*b)) {
            // 1^k equals 1
            (Node::Num { ref val, .. }, _) if val.is_one() => Node::one(),

            // k^0 equals 1
            (_, Node::Num { ref val, .. }) if val.is_zero() => Node::one(),

            // we cannot simplify
            (a, b) => Node::Exp(Box::new(a), Box::new(b)),
        },

        Node::Inverse(a) => match simplify(*a) {
            Node::Num { val, input_base } => {
                let (numer, denom) = val.into();

                Node::Num {
                    // take the inverse by swapping numerator and denominator
                    val: BigRational::new(denom, numer),
                    input_base,
                }
            },

            // cannot simplify
            node => Node::Inverse(Box::new(node)),
        },

        Node::Sin(ref inner) | Node::Cos(ref inner) | Node::Tan(ref inner) => {
            let inner_simplified = simplify(*inner.clone());
            if let Some(mut pi_multiplier) = get_pi_multiplier(&inner_simplified) {
                // simplify (2a + b)pi as b*pi with -1 <= b <= 1
                pi_multiplier %= 2;
                // Map negative b's to positive, but keep the same result in
                // the end.
                if pi_multiplier < 0 {
                    pi_multiplier += 2;
                }
                return match pi_multiplier {
                    0 => match &node {
                        Node::Sin(_) | Node::Tan(_) => Node::zero(),
                        Node::Cos(_) => Node::one(),
                        _ => unreachable!(),
                    },
                    1 => match &node {
                        Node::Sin(_) => Node::one(),
                        Node::Cos(_) => Node::minus_one(),
                        Node::Tan(_) => Node::zero(),
                        _ => unreachable!(),
                    }
                    _ => unreachable!(),
                };
            }
            node
        }

        // fallback to doing nothing
        _ => node,
    }
}

fn get_pi_multiplier(node: &Node) -> Option<i64> {
    match node {
        Node::Const(ConstKind::Pi) => Some(1),
        Node::Const(ConstKind::Tau) => Some(2),
        Node::Num { val, .. } if val.is_zero() => Some(0),
        Node::VarOp { children, kind: VarOpKind::Mul } => {
            let mut multiplier: i64 = 1;
            let mut has_pi = false;
            for c in children {
                if let Node::Num { val, .. } = c {
                    if !val.denom().is_one() {
                        // no support for fractional multipliers
                        return None;
                    }
                    let new = val.numer().to_i64()
                        .and_then(|x| multiplier.checked_mul(x));
                    match new {
                        Some(x) => multiplier = x,
                        // overflow error
                        None => return None,
                    }
                } else if let Some(m) = get_pi_multiplier(&c) {
                    if m == 0 {
                        // zero times anything is zero
                        return Some(0);
                    }
                    if has_pi {
                        // We already have pi, so this will be pi^2 which we
                        // do not support here because we're supposed to
                        // return the multiplier of pi as an integer.
                        return None;
                    }
                    let new = multiplier.checked_mul(m);
                    match new {
                        Some(x) => {
                            multiplier = x;
                            has_pi = true;
                        }
                        // overflow error
                        None => return None,
                    }
                } else {
                    // complex node that we do not understand
                    return None;
                }
            }
            if has_pi {
                Some(multiplier)
            } else {
                None
            }
        },
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
            }).collect::<Vec<_>>();
    }

    result
}
