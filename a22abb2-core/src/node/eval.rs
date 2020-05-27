use std::f64::consts::{E, PI};
use std::ops::*;

use crate::ratio2flt::ratio_to_f64;
use crate::EvalResult;
use super::{ConstKind, Node};
use super::util::{fold_nodes, get_op_result_base};

/// Approximates the node value.
pub fn eval(node: &Node) -> EvalResult {
    match node {
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
            display_base: *input_base,
        },
        Node::Sum(children) => fold_nodes(children.iter(), 0.0, Add::add),
        Node::Product(children) => fold_nodes(children.iter(), 1.0, Mul::mul),
        Node::Exp(a, b) => {
            let a = eval(a);
            let b = eval(b);
            if b.val == 0.0 {
                return EvalResult {
                    val: 1.0,
                    display_base: a.display_base,
                };
            } else if b.val == 1.0 {
                return EvalResult {
                    val: a.val,
                    display_base: a.display_base,
                };
            } else if b.val == -1.0 {
                return EvalResult {
                    val: 1.0 / a.val,
                    display_base: a.display_base,
                };
            }
            EvalResult {
                val: a.val.powf(b.val),
                display_base: get_op_result_base(a.display_base, b.display_base),
            }
        }
        Node::Sin(inner) => eval_map(inner, f64::sin, false),
        Node::Cos(inner) => eval_map(inner, f64::cos, false),
        Node::Tan(inner) => eval_map(inner, f64::tan, false),
    }
}

fn eval_map<F: Fn(f64) -> f64>(node: &Node, f: F, keep_base: bool) -> EvalResult {
    let original = eval(node);
    EvalResult {
        val: f(original.val),
        display_base: if keep_base {
            original.display_base
        } else {
            None
        },
    }
}
