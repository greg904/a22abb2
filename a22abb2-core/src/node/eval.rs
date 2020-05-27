use num_traits::Float;
use std::f64::consts::{E, PI};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::*;

use crate::ratio2flt::ratio_to_f64;
use super::{ConstKind, Node};
use super::util::{fold_nodes, get_op_result_base};

/// A struct that holds the result of a calculation.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct EvalSuccess {
    /// The result value
    pub val: f64,

    /// The base the result value should be displayed in
    pub display_base: Option<u32>,
}

/// A description of the error of a calculation.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum EvalError {
    ZeroToThePowerOfZero,
}

/// Approximates the node value.
pub fn eval(node: &Node) -> Result<EvalSuccess, EvalError> {
    Ok(match node {
        Node::Const(kind) => EvalSuccess {
            val: match kind {
                ConstKind::Pi => PI,
                ConstKind::Tau => PI * 2.0,
                ConstKind::E => E,
            },
            display_base: None,
        },
        Node::Num { val, input_base } => EvalSuccess {
            val: ratio_to_f64(&val),
            display_base: *input_base,
        },
        Node::Sum(children) => fold_nodes(children.iter(), 0.0, Add::add)?,
        Node::Product(children) => fold_nodes(children.iter(), 1.0, Mul::mul)?,
        Node::Exp(a, b) => {
            let a = eval(a)?;
            let b = eval(b)?;
            if b.val == 0.0 {
                if a.val == 0.0 {
                    return Err(EvalError::ZeroToThePowerOfZero);
                }
                return Ok(EvalSuccess {
                    val: 1.0,
                    display_base: a.display_base,
                });
            } else if b.val == 1.0 {
                return Ok(EvalSuccess {
                    val: a.val,
                    display_base: a.display_base,
                });
            } else if b.val == -1.0 {
                return Ok(EvalSuccess {
                    val: 1.0 / a.val,
                    display_base: a.display_base,
                });
            }
            EvalSuccess {
                val: a.val.powf(b.val),
                display_base: get_op_result_base(a.display_base, b.display_base),
            }
        }
        Node::Sin(inner) => eval_map(inner, f64::sin, false)?,
        Node::Cos(inner) => eval_map(inner, f64::cos, false)?,
        Node::Tan(inner) => eval_map(inner, f64::tan, false)?,
    })
}

fn eval_map<F: Fn(f64) -> f64>(node: &Node, f: F, keep_base: bool) -> Result<EvalSuccess, EvalError> {
    let original = eval(node)?;
    Ok(EvalSuccess {
        val: f(original.val),
        display_base: if keep_base {
            original.display_base
        } else {
            None
        },
    })
}

impl Display for EvalSuccess {
    fn fmt(&self, out: &mut Formatter) -> fmt::Result {
        match self.display_base.unwrap_or(10) {
            10 => self.val.fmt(out),

            // TODO: add tests
            2 => {
                let (mantissa, exp, sign) = self.val.integer_decode();
                let is_nonnegative = sign == 1 || mantissa == 0;
                let mut result = String::new();
                let mut is_printing = true;
                let mut consecutive_zeros = 0;

                if exp <= -53 {
                    result += "0.";
                    for _ in exp..-53 {
                        result.push('0');
                    }
                }
                for i in (0..53).rev() {
                    let bit = mantissa & (1 << i) != 0;
                    let bit_exp = exp + i;
                    if bit_exp < 0 {
                        // check how many digits we must print after the decimal separator
                        if let Some(precision) = out.precision() {
                            if (-bit_exp as usize) > precision {
                                break;
                            }
                        }
                    }
                    // do not print leading zeros
                    if bit || bit_exp < 0 {
                        is_printing = true;
                    }
                    if is_printing {
                        if bit_exp < 0 && !bit && out.precision().is_none() {
                            // Do not print zeros after the decimal separator
                            // unless there is a one after.
                            // Example: 0b1.100000000 -> 0b1.1
                            consecutive_zeros += 1;
                        } else {
                            if consecutive_zeros > 0 {
                                // we forgot to print the decimal separator
                                if bit_exp + consecutive_zeros == -1 {
                                    result.push('.');
                                }
                                for _ in 0..consecutive_zeros {
                                    result.push('0');
                                }
                                consecutive_zeros = 0;
                            }
                            if i != 52 && bit_exp == -1 {
                                // we got our first digit after a dot!
                                result.push('.');
                            }
                            result.push(if bit { '0' } else { '1' });
                        }
                    }
                }
                if let Some(precision) = out.precision() {
                    if exp < 0 {
                        let already_printed = -exp as usize;

                        // fill with zeros at the end
                        for _ in already_printed..precision {
                            result.push('0');
                        }
                    }
                }

                out.pad_integral(is_nonnegative, "0b", &result)
            }
            _ => {
                eprintln!("cannot print result in bases other than 10 and two yet");
                self.val.fmt(out)
            },
        }
    }
}
