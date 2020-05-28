extern crate a22abb2_core;
extern crate num_rational;
extern crate num_traits;
extern crate rand;

use a22abb2_core::node::*;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rand::prelude::*;

struct RecursiveCtx {
    pub depth: u32,
    pub inside_trigo: bool,
    pub inside_exp: u32,
    pub inside_sum: u32,
    pub inside_product: u32,
}

fn random_const(ctx: &RecursiveCtx) -> ConstKind {
    if ctx.inside_exp > 0 {
        // tau is still too big!
        // (tau^tau is approximately 103540.920434)
        return if thread_rng().gen() {
            ConstKind::Pi
        } else {
            ConstKind::E
        };
    }
    const CONSTS: [ConstKind; 3] = [ConstKind::Pi, ConstKind::Tau, ConstKind::E];
    *CONSTS.choose(&mut thread_rng()).unwrap()
}

fn random_num(ctx: &RecursiveCtx) -> BigRational {
    let mut rng = thread_rng();
    match rng.gen_range(0, 10) {
        0 => Zero::zero(),
        1 => One::one(),
        x => {
            // If we're in a trigonometry function, we don't want enormous
            // numbers.
            if !ctx.inside_trigo && x == 9 {
                return BigRational::from_integer(rng.gen::<i32>().into());
            }
            // If we're in a exponential function, we really don't want big
            // numbers.
            if ctx.inside_exp > 0 {
                return BigRational::from_integer(rng.gen_range(-3, 4).into());
            }
            BigRational::from_integer(rng.gen_range(-100, 100).into())
        }
    }
}

fn random_base() -> Option<u32> {
    let mut rng = thread_rng();
    match rng.gen_range(0, 10) {
        0 => None,
        1 => Some(2),
        2 => Some(8),
        3..=8 => Some(10),
        9 => Some(16),
        _ => unreachable!(),
    }
}

fn random_vararg_op(ctx: RecursiveCtx, is_sum: bool) -> Node {
    let mut rng = thread_rng();

    let mut children = Vec::new();
    let count = if ctx.inside_exp == 0 {
        rng.gen_range(2, 7)
    } else {
        // prevent big numbers
        2
    };
    for _ in 0..count {
        children.push(random_node(ctx.for_vararg_op(is_sum)));
    }

    return if is_sum {
        Node::Sum(children)
    } else {
        Node::Product(children)
    };
}

fn random_node(ctx: RecursiveCtx) -> Node {
    let mut rng = thread_rng();
    // limit the amount of node depth
    if ctx.depth < 5 && rng.gen_range(0, 10) > 2 {
        // pick a composite node
        match rng.gen_range(0, 6) {
            0 if ctx.inside_sum < 3 => return random_vararg_op(ctx, true),
            1 if ctx.inside_product < 3 => return random_vararg_op(ctx, false),
            2 if ctx.inside_exp < 2 => return Node::Exp(
                Box::new(random_node(ctx.for_exp_call())),
                Box::new(random_node(ctx.for_exp_call())),
            ),
            3 => return Node::Sin(Box::new(random_node(ctx.for_trigo_call()))),
            4 => return Node::Cos(Box::new(random_node(ctx.for_trigo_call()))),
            5 => return Node::Tan(Box::new(random_node(ctx.for_trigo_call()))),
            _ => {},
        }
    }
    // pick leaf node
    match rng.gen_range(0, 2) {
        0 => Node::Const(random_const(&ctx)),
        1 => Node::Num {
            val: random_num(&ctx),
            input_base: random_base(),
        },
        _ => unreachable!(),
    }
}

impl RecursiveCtx {
    fn new() -> Self {
        Self {
            depth: 0,
            inside_trigo: false,
            inside_exp: 0,
            inside_sum: 0,
            inside_product: 0,
        }
    }

    fn for_trigo_call(&self) -> Self {
        Self {
            depth: self.depth + 1,
            inside_trigo: true,
            inside_exp: self.inside_exp,
            inside_sum: self.inside_sum,
            inside_product: self.inside_product,
        }
    }

    fn for_exp_call(&self) -> Self {
        Self {
            depth: self.depth + 1,
            inside_trigo: self.inside_trigo,
            inside_exp: self.inside_exp + 1,
            inside_sum: self.inside_sum,
            inside_product: self.inside_product,
        }
    }

    fn for_vararg_op(&self, is_sum: bool) -> Self {
        Self {
            depth: self.depth + 1,
            inside_trigo: self.inside_trigo,
            inside_exp: self.inside_exp,
            inside_sum: if is_sum { self.inside_sum + 1 } else { self.inside_sum },
            inside_product: if !is_sum { self.inside_product + 1 } else { self.inside_product },
        }
    }
}

fn main() {
    for i in 0..5000 {
        let node = random_node(RecursiveCtx::new());

        if i != 0 {
            println!("");
        }
        println!("Testing {}:\n", node);

        if let Ok(ground_truth) = node.eval() {
            println!("- eval before simplification: {}", ground_truth);

            // will cause precision errors and other weird errors
            if !ground_truth.val.is_finite() {
                continue;
            }

            if let Ok(simplified) = node.simplify() {
                println!("- simplified expression: {}", simplified);

                if let Ok(simplified_result) = simplified.eval() {
                    println!("- eval after simplification: {}", simplified_result.val);

                    let mut is_equal = false;
                    is_equal |= (simplified_result.val - ground_truth.val).abs() < 0.1;
                    if ground_truth.val != 0.0 {
                        let rel_error = ((simplified_result.val - ground_truth.val) / ground_truth.val).abs();
                        is_equal |= rel_error < 0.1;
                    }
                    assert!(is_equal);
                    // TODO: uncomment
                    // assert_eq!(simplified_result.display_base, ground_truth.display_base);
                } else {
                    println!("- eval after simplification: (error)");
                }
            } else {
                println!("- simplified expression: (error)");
            }
        } else {
            println!("- eval before simplification: (error)");
        }
    }
}
