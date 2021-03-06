//! A framework for writing generative models in the rust programming language. 
/*!
# Example of Importance Sampling
```rust
use r_gen::{sample, r_gen}; 
use r_gen::{simulate, generate, distributions::{Value, Distribution}, trace::{Choicemap, Trace}}; 
use std::rc::Rc;
fn main() {
    //Define our generative model. 
    #[r_gen]
    fn my_model(():()){
        let p = sample!(format!("p"), Distribution::Beta(1.0, 1.0)); 
        sample!(format!("num_heads"), Distribution::Binomial(100, p.into()));
    }

    //Run the model once in the forward direction and record the observations. 
    let (t, _) : (Trace, _)= simulate(&mut my_model, ());
    let choices = Choicemap::from(vec![("num_heads", t.choices["num_heads"].clone())]);

    //Perform importance resampling to get an estimate for the value of p. 
    let mut traces = Vec::new();
    for _ in 0..1000 {
        let (gt, _) : (Trace, _)= generate(&mut my_model, (), &choices);
        traces.push(gt); 
    }
    
    println!("Actual value for p:\t {}", t.choices["p"]); 
    println!("Generated value for p:\t {}", Trace::sample_weighted_traces(&traces).unwrap().choices["p"]); 
}
```
Outputs:
```shell
Actual value for p:      0.8011431168181488
Generated value for p:   0.7879998086169554
```
*/

#![warn(missing_docs)]

#[allow(unused_attributes)]
#[macro_use]
pub use r_gen_macro::r_gen; 

#[allow(unused_imports)]
#[macro_use]
extern crate r_gen_macro;

/**
The macro that is used for sampling from a distribution. 
# Example
```
use r_gen::{sample, r_gen}; 
use r_gen::{simulate, distributions::{Value, Distribution}, trace::{Choicemap, Trace}}; 
use std::rc::Rc;

#[r_gen]
fn my_model(():()) {
    let p = sample!(format!("p"), Distribution::Bernoulli(0.5)); 
    print!("p: {}", p);
}
simulate(&mut my_model, ()); 
``` 
Takes the form: identifier,  Distribution. The identifier will have the value sampled from the distribution stored in
it. It can be used later. p will have type ```Value```.
# Example (Store results in an array)
```
use r_gen::{sample, r_gen}; 
use r_gen::{simulate, distributions::{Value, Distribution}, trace::{Choicemap, Trace}}; 
use std::rc::Rc;

#[r_gen] 
fn flip_my_biased_coins((n, p) : (usize, f64)) {
    let mut flips = vec![Value::Integer(0); n]; 
    for i in 0..n {
        flips[i] = sample!(format!("flip_{}", i), Distribution::Bernoulli(p)); 
    }
}
let (tr, _) = simulate(&mut flip_my_biased_coins, (10, 0.5));
println!("{}", tr.get_trace_string()); 
```
*/
#[macro_export]
macro_rules! sample {
    ($sample_ident:ident $trace_ident:ident $name:expr, $dist:expr) => (
        (Rc::get_mut(&mut $sample_ident).unwrap())(&$name, $dist, $trace_ident);
    );
}


#[allow(unused_variables)]
#[cfg(test)]
mod tests { 
    use std::rc::Rc;

    use crate::{trace::{Trace, Choicemap}, distributions::{Distribution, Value}};
    use crate::{simulate, generate}; 

    #[test]
    fn test_simulate(){
        fn flip_biased_coin(mut sample : Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, trace : &mut Trace, p : f64) {
            // flip ~ Bernoulli(p)
            let flip = (Rc::get_mut(&mut sample).unwrap())(&String::from("flip"), Distribution::Bernoulli(p), trace);
        }
        let (t, _) : (Trace, _)= simulate(&mut flip_biased_coin, 0.2); 
        println!("test_simulate flip_biased_coin trace: {:?}", t); 

        fn flip_multiple_biased_coins(mut sample : Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, trace : &mut Trace, (n, p) : (i64, f64)) {
            // heads ~ Bernoulli(p)
            let heads = (Rc::get_mut(&mut sample).unwrap())(&String::from("heads"), Distribution::Binomial(n, p), trace);
        }
        let (t, _) : (Trace, _)= simulate(&mut flip_multiple_biased_coins, (5, 0.7)); 
        println!("test_simulate flip_multiple_biased_coin trace: {:?}", t);
    }

    #[test]
    fn test_generate(){
        #[r_gen]
        fn flip_multiple_biased_coins((n, p) : (i64, f64)) {
            let heads = sample!(format!("heads"), Distribution::Binomial(n, p)); 
            println!("Result of flips: {:?}", heads)
        }
        let mut constraints = Choicemap::new(); 
        constraints.add_choice("heads", Value::Integer(4)); 
        let (trace, _) : (Trace, _)= generate(&mut flip_multiple_biased_coins, (5, 0.7), &constraints); 
        println!("Trace from generate: {:?}", trace);
    }

    #[test]
    fn test_macros(){
        #[r_gen]
        fn my_coin_model(p : f64) {
            let flip = sample!(format!("flip"), Distribution::Bernoulli(p)); 
            println!("Result of flip: {:?}", flip)
        }
        let (trace, _) = simulate(&mut my_coin_model, 0.2); 
        println!("testing macro: {:?}", trace);


        #[r_gen]
        fn flip_multiple_biased_coins((n, p) : (i64, f64)) {
            let heads = sample!(format!("heads"), Distribution::Binomial(n, p)); 
            println!("Result of flips: {:?}", heads)
        }
        let (trace, _) : (Trace, _)= simulate(&mut flip_multiple_biased_coins, (5, 0.7)); 
        println!("tesing macro: {:?}", trace);

        #[r_gen] 
        fn flip_my_biased_coin((n, p) : (usize, f64)) {
            let mut flips = vec![Value::Integer(0); n]; 
            for i in 0..n {
                flips[i] = sample!(format!("flips_{}", i), Distribution::Bernoulli(p)); 
            }
        }
        let (trace, _) : (Trace, _)= simulate(&mut flip_my_biased_coin, (5 as usize, 0.7)); 
        println!("my flip coin trace: {:?}", trace);

        #[r_gen] 
        fn flip_my_biased_coin2((n, p) : (usize, f64)) {
            let mut flips = vec![Value::Integer(0); n]; 
            for i in 0..n {
                flips[i] = sample!(format!("flips_{}", i), Distribution::Bernoulli(p)); 
            }
            println!("flips: {:?}", flips); 
        }
        let (trace, _) : (Trace, _)= simulate(&mut flip_my_biased_coin2, (5 as usize, 0.7)); 
        println!("my flip coin 2 trace: {:?}", trace);
    }

    #[test]
    fn test_bernoulli(){
        #[r_gen]
        fn my_bernoulli(p : f64) {
            let mut tests = vec![Value::Real(0.0); 100]; 
            for i in 0..100 {
                tests[i] = sample!(format!("tests_{}", i), Distribution::Bernoulli(p));
            }
            let mut tot : f64 = 0.0; 
            for t in tests {
                match t {
                    Value::Boolean(true) => {
                        tot = tot + 1.0; 
                    }, 
                    _ => ()
                }
            }
            println!("P: {}\nResult of tests:{:?}", p, tot/100.0); 
        }
        let (_, _) = simulate(&mut my_bernoulli, 0.5); 
    }

    #[test]
    fn test_binom(){
        #[r_gen]
        fn my_binomial((n, p): (i64, f64)) {
            let mut tests = vec![Value::Real(0.0); 100]; 
            for i in 0..100 {
                tests[i] = sample!(format!("tests_{}", i), Distribution::Binomial(n, p));
            }
            let mut tot : f64 = 0.0; 
            for t in tests {
                match t {
                    Value::Integer(i) => {
                        tot = tot + (i as f64); 
                    }, 
                    _ => ()
                }
            }
            println!("N*P: {}\nResult of tests:{:?}", ((n as f64)*p), tot/100.0); 
        }
        let (_, _) = simulate(&mut my_binomial, (100, 0.5)); 
    }

    #[test]
    fn test_normal(){
        #[r_gen]
        fn my_normal((m, s): (f64, f64)) {
            let mut tests = vec![Value::Real(0.0); 100]; 
            for i in 0..100 {
                tests[i] = sample!(format!("tests_{}", i), Distribution::Normal(m, s));
            }
            let mut tot : f64 = 0.0; 
            for t in tests {
                match t {
                    Value::Real(r) => {
                        tot = tot + r;
                    },
                    _ => ()
                }
            }
            println!("Mean: {}\nResult of tests:{:?}", m, tot/100.0); 
        }
        let (_, _) = simulate(&mut my_normal, (60.0, 10.0)); 
    }

    #[test]
    fn test_importance_resampling(){
        //Define our generative model. 
        #[r_gen]
        fn my_model(():()){
            let p = sample!(format!("p"), Distribution::Beta(1.0, 1.0)); 
            sample!(format!("num_heads"), Distribution::Binomial(100, p.into()));
        }

        //Run the model once in the forward direction and record the observations. 
        let (t, _) : (Trace, _)= simulate(&mut my_model, ());
        let choices = Choicemap::from(vec![("num_heads", t.choices["num_heads"].clone())]);

        //Perform importance resampling to get an estimate for the value of p. 
        let mut traces = Vec::new();
        for _ in 0..1000 {
            let (gt, _) : (Trace, _)= generate(&mut my_model, (), &choices);
            traces.push(gt); 
        }
        
        println!("Actual value for p:\t {}", t.choices["p"]); 
        println!("Generated value for p:\t {}", Trace::sample_weighted_traces(&traces).unwrap().choices["p"]); 
    }

    #[test] 
    fn test_trace_string(){
        #[r_gen]
        fn my_biased_coin_model(():()){
            let p = sample!(format!("p"), Distribution::Beta(1.0, 1.0));            //Sample p from a uniform. 
            sample!(format!("num_heads"), Distribution::Binomial(100, p.into()));   //Flip 100 coins where P(Heads)=p
        }
        println!("GO"); 
        let (trace, result) = simulate(&mut my_biased_coin_model, ()); 
        println!("Trace String: \n{}", trace.get_trace_string());
    }

}


use std::rc::Rc;

use self::{distributions::{Value, Sampleable}, trace::{Choicemap, Trace}};

//Re-export the other sub modules. 
pub mod distributions; 
pub mod trace; 

/**
Run the given generative model in the forward direction.
As input, it takes a generative model (function with the #[r_gen] tag) and the arguments to that function. 
Returns a tuple of the trace generated by running the function and the return value of the function itself.
# Example
```
use r_gen::{sample, r_gen}; 
use r_gen::{simulate, distributions::{Value, Distribution}, trace::{Choicemap, Trace}}; 
use std::rc::Rc;
#[r_gen]
fn my_biased_coin_model(():()){
    let p = sample!(format!("p"), Distribution::Beta(1.0, 1.0));            //Sample p from a uniform. 
    sample!(format!("num_heads"), Distribution::Binomial(100, p.into()));   //Flip 100 coins where P(Heads)=p
}
println!("GO"); 
let (trace, result) = simulate(&mut my_biased_coin_model, ()); 
println!("Trace String: \n{}", trace.get_trace_string());
```
Outputs: 
```shell
Trace String: 
num_heads => 37
p => 0.38724904991570935
```
*/
pub fn simulate<F, A, R, S : Sampleable>(generative_function : &mut F, arguments : A) -> (Trace, R) 
where 
F : FnMut(Rc<dyn FnMut(&String, S, &mut Trace) -> Value>, &mut Trace, A) -> R, 
{
    let sample = |name : &String, dist : S, trace : &mut Trace| {
        let value = dist.sample();                              //Sample a value. 
        let prob = dist.liklihood(&value).unwrap();               //Compute the probability of this value. 
        trace.update_logscore(prob);                         //Update the log score with the pdf. 
        trace.choices.add_choice(&name, value.clone());     //Add the choice to the hashmap.
        value
    }; 
    let mut trace = Trace::new(); 
    let return_value = generative_function(Rc::new(sample), &mut trace, arguments); 
    (trace, return_value)
}

/**
Run a generative model in the forward direction, fixing certian decisions or observations.
As input, it takes a generative model (function with the #[r_gen] tag), the arguments to that function, and a choicemap of the observed variables. 
Returns a tuple of the trace generated by running the function and the return value of the function itself.
# Example
```
use r_gen::{sample, r_gen}; 
use r_gen::{generate, distributions::{Value, Distribution}, trace::{Choicemap, Trace}}; 
use std::rc::Rc;
#[r_gen]
fn my_biased_coin_model(():()){
    let p = sample!(format!("p"), Distribution::Beta(1.0, 1.0));            //Sample p from a uniform. 
    sample!(format!("num_heads"), Distribution::Binomial(100, p.into()));   //Flip 100 coins where P(Heads)=p
}
let choices = Choicemap::from(vec![("p", Value::Real(0.1))]);     //Fix the value p=0.1
let (trace, result) = generate(&mut my_biased_coin_model, (), &choices); 
```
*/
pub fn generate<F, A, R, S: Sampleable>(generative_function : &mut F, arguments : A, conditions : &Choicemap) -> (Trace, R) 
where 
F : FnMut(Rc<dyn FnMut(&String, S, &mut Trace) -> Value>, &mut Trace, A) -> R, 
{   
    let sample = |name : &String, dist : S, trace : &mut Trace| {
        let mut _value = Value::Real(0.0); 
        _value = if trace.choices.contains_key(name) {
            trace.choices[name.as_str()].clone()
        } else {
            dist.sample()
        };
        let prob = dist.liklihood(&_value).unwrap();                    //Compute the probability of this value. 
        trace.update_logscore(prob);                               //Update the log score with the pdf. 
        trace.choices.add_choice(name.as_str(), _value.clone());  //Add the choice to the hashmap.
        _value 
    };
    let mut trace = Trace::new(); 
    for (k, v) in conditions.get_choices() {
        trace.choices.add_choice(k, v); 
    }
    let return_value = generative_function(Rc::new(sample), &mut trace, arguments);
    (trace, return_value)
}