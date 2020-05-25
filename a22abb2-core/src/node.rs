use super::ratio2flt::ratio_to_f64;
use super::EvalResult;

use std::collections::HashMap;
use std::f64::consts::{E, PI};
use std::fmt;
use std::iter;
use std::ops::{Add, Mul};

use num_rational::BigRational;
use num_traits::{One, Zero};

use either::Either;

/// A constant in mathematics
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ConstKind {
    Pi,
    Tau,
    E,
}

/// A kind of operator that can take multiple children
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum VarOpKind {
    Add,
    Mul,
}

impl VarOpKind {
    pub fn identity_f64(self) -> f64 {
        match self {
            VarOpKind::Add => 0.0,
            VarOpKind::Mul => 1.0,
        }
    }

    pub fn identity_bigr(self) -> BigRational {
        match self {
            VarOpKind::Add => Zero::zero(),
            VarOpKind::Mul => One::one(),
        }
    }

    pub fn eval_f64_fn(self) -> &'static dyn Fn(f64, f64) -> f64 {
        match self {
            VarOpKind::Add => &Add::add,
            VarOpKind::Mul => &Mul::mul,
        }
    }

    pub fn eval_bigr_fn(self) -> &'static dyn Fn(BigRational, BigRational) -> BigRational {
        match self {
            VarOpKind::Add => &Add::add,
            VarOpKind::Mul => &Mul::mul,
        }
    }

    fn compress(self, node: Node, count: Node) -> Node {
        match self {
            VarOpKind::Add => Node::mul(node, count),
            VarOpKind::Mul => Node::Exp(Box::new(node), Box::new(count)),
        }
    }
}

/// A node is an operation in the AST (abstract syntax tree).
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Node {
    Const(ConstKind),
    Num {
        /// The number the node represents
        val: BigRational,

        /// The base the number was written in by the user, if it was written
        /// by the user
        input_base: Option<u32>,
    },
    Inverse(Box<Node>),
    VarOp {
        kind: VarOpKind,
        children: Vec<Node>,
    },
    Exp(Box<Node>, Box<Node>),
}

impl Node {
    /// Approximates the node value.
    pub fn eval(self) -> EvalResult {
        match self {
            Node::Const(kind) => EvalResult {
                val: match kind {
                    ConstKind::Pi => PI,
                    ConstKind::Tau => PI * 2.0,
                    ConstKind::E => E,
                },
                display_base: None,
            },
            Node::Num { val, input_base } => EvalResult {
                val: ratio_to_f64(&val),
                display_base: input_base,
            },
            Node::Inverse(inner) => {
                let inner = inner.eval();
                EvalResult {
                    val: 1.0 / inner.val,
                    display_base: inner.display_base,
                }
            }
            Node::VarOp { kind, children } => Node::eval_var_op(children.into_iter(), kind),
            Node::Exp(a, b) => {
                let a = a.eval();
                let b = b.eval();
                EvalResult {
                    val: a.val.powf(b.val),
                    display_base: Node::get_op_result_base(a.display_base, b.display_base),
                }
            }
        }
    }

    fn eval_var_op<I>(children: I, kind: VarOpKind) -> EvalResult
    where
        I: Iterator<Item = Node>,
    {
        let mut result = kind.identity_f64();
        let mut result_base = None;

        for child in children {
            let child = child.eval();
            result = kind.eval_f64_fn()(result, child.val);
            result_base = Node::get_op_result_base(result_base, child.display_base);
        }

        EvalResult {
            val: result,
            display_base: result_base,
        }
    }

    fn get_op_result_base(a_base: Option<u32>, b_base: Option<u32>) -> Option<u32> {
        match (a_base, b_base) {
            (Some(val), None) | (None, Some(val)) => Some(val),

            // prefer the more interesting bases
            (Some(10), Some(other)) | (Some(other), Some(10)) => Some(other),
            (Some(2), _) | (_, Some(2)) => Some(2),

            (Some(a), Some(_)) => Some(a), // prefer the base of the first term
            (None, None) => None,
        }
    }

    // TODO: write documentation
    pub fn deep_reduce(self) -> Node {
        match self {
            Node::VarOp { kind, children } => {
                let children = Node::deep_flatten_children(children, kind)
                    .into_iter()
                    .map(|t| t.deep_reduce());
                let children = Node::collapse_numbers(children, kind);

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
                        let factors = Node::collapse_numbers(factors.into_iter(), VarOpKind::Add);

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

            Node::Exp(a, b) => match (a.deep_reduce(), b.deep_reduce()) {
                // 1^k equals 1
                (Node::Num { ref val, .. }, _) if val.is_one() => Node::one(),

                // k^0 equals 1
                (_, Node::Num { ref val, .. }) if val.is_zero() => Node::one(),

                // we cannot simplify
                (a, b) => Node::Exp(Box::new(a), Box::new(b)),
            },

            Node::Inverse(a) => match a.deep_reduce() {
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

            // fallback to doing nothing
            node => node,
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

    pub fn one() -> Node {
        Node::Num {
            val: One::one(),
            input_base: None,
        }
    }

    pub fn minus_one() -> Node {
        Node::Num {
            val: -BigRational::one(),
            input_base: None,
        }
    }

    pub fn add(a: Node, b: Node) -> Node {
        Node::op(VarOpKind::Add, a, b)
    }

    pub fn sub(a: Node, b: Node) -> Node {
        Node::add(a, Node::opposite(b))
    }

    pub fn mul(a: Node, b: Node) -> Node {
        Node::op(VarOpKind::Mul, a, b)
    }

    pub fn div(a: Node, b: Node) -> Node {
        Node::mul(a, Node::Inverse(Box::new(b)))
    }

    fn op(kind: VarOpKind, a: Node, b: Node) -> Node {
        Node::VarOp {
            kind,
            children: vec![a, b],
        }
    }

    pub fn opposite(inner: Node) -> Node {
        Node::mul(Node::minus_one(), inner)
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Const(kind) => match kind {
                ConstKind::Pi => write!(f, "pi"),
                ConstKind::Tau => write!(f, "tau"),
                ConstKind::E => write!(f, "e"),
            },
            // TODO: print in correct base
            Node::Num { val, input_base: _ } => write!(f, "{}", val),

            Node::Inverse(inner) => write!(f, "1/{}", inner),
            Node::VarOp { kind, children } => {
                let mut first = true;
                for c in children {
                    if first {
                        first = false;
                    } else {
                        let op_char = match kind {
                            VarOpKind::Add => '+',
                            VarOpKind::Mul => '*',
                        };
                        write!(f, " {} ", op_char)?;
                    }
                    write!(f, "{}", c)?;
                }
                Ok(())
            },
            Node::Exp(a, b) => write!(f, "{}^{}", a, b),
        }
    }
}
