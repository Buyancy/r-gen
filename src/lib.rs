#[macro_use]
extern crate r_gen_macro;

//The macro to convert the sample statement. 
#[macro_export]
macro_rules! sample {

    ($sample_ident:ident $trace_ident:ident $name:ident ~ $dist:expr) => (
        let $name = (Rc::get_mut(&mut $sample_ident).unwrap())(&String::from(stringify!($name)), $dist, $trace_ident);
    );

    ($sample_ident:ident $trace_ident:ident $name:ident => $i:ident ~ $dist:expr) => (
        let mut s = String::from(stringify!($name));
        s.push_str("[");
        s.push_str(&$i.to_string());
        s.push_str("]");
        $name[$i] = (Rc::get_mut(&mut $sample_ident).unwrap())(&s, $dist, $trace_ident);
    );

    ($sample_ident:ident $trace_ident:ident $name:ident [ $i:ident ] ~ $dist:expr) => (
        let mut s = String::from(stringify!($name));
        s.push_str("[");
        s.push_str(&$i.to_string());
        s.push_str("]");
        $name[$i] = (Rc::get_mut(&mut $sample_ident).unwrap())(&s, $dist, $trace_ident);
    );

}

#[cfg(test)]
mod tests { 
   

    #[test]
    fn test_macro() {
        assert_eq!(2+2, 4);
    }

    use std::collections::HashMap;
    use std::rc::Rc;
    use crate::r_gen::{Trace, Value};
    use crate::r_gen::distributions::Distribution; 
    use crate::r_gen::{simulate, generate}; 
    #[test]
    fn test_simulate(){
        fn flip_biased_coin(mut sample : Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, trace : &mut Trace, p : f64) {
            // flip ~ Bernoulli(p)
            (Rc::get_mut(&mut sample).unwrap())(&String::from("flip"), Distribution::Bernoulli(p), trace);
        }
        let (t, _) : (Trace, _)= simulate(&mut flip_biased_coin, 0.2); 
        println!("test_simulate flip_biased_coin trace: {:?}", t); 

        fn flip_multiple_biased_coins(mut sample : Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, trace : &mut Trace, (n, p) : (i64, f64)) {
            // heads ~ Bernoulli(p)
            (Rc::get_mut(&mut sample).unwrap())(&String::from("heads"), Distribution::Binomial(n, p), trace);
        }
        let (t, _) : (Trace, _)= simulate(&mut flip_multiple_biased_coins, (5, 0.7)); 
        println!("test_simulate flip_multiple_biased_coin trace: {:?}", t);
    }

    #[test]
    fn test_generate(){
        #[r_gen]
        fn flip_multiple_biased_coins((n, p) : (i64, f64)) {
            sample!(heads ~ Distribution::Binomial(n, p)); 
            println!("Result of flips: {:?}", heads)
        }
        let mut constraints = HashMap::new(); 
        constraints.insert(String::from("heads"), Value::Integer(4)); 
        let (trace, _) : (Trace, _)= generate(&mut flip_multiple_biased_coins, (5, 0.7), &constraints); 
        println!("Trace from generate: {:?}", trace);
    }

    #[test]
    fn test_macros(){
        #[r_gen]
        fn my_coin_model(p : f64) {
            sample!(flip ~ Distribution::Bernoulli(p)); 
            println!("Result of flip: {:?}", flip)
        }
        let (trace, _) = simulate(&mut my_coin_model, 0.2); 
        println!("testing macro: {:?}", trace);


        #[r_gen]
        fn flip_multiple_biased_coins((n, p) : (i64, f64)) {
            sample!(heads ~ Distribution::Binomial(n, p)); 
            println!("Result of flips: {:?}", heads)
        }
        let (trace, _) : (Trace, _)= simulate(&mut flip_multiple_biased_coins, (5, 0.7)); 
        println!("tesing macro: {:?}", trace);

        #[r_gen] 
        fn flip_my_biased_coin((n, p) : (usize, f64)) {
            let mut flips = vec![Value::Integer(0); n]; 
            for i in 0..n {
                sample!(flips => i ~ Distribution::Bernoulli(p)); 
            }
        }
        let (trace, _) : (Trace, _)= simulate(&mut flip_my_biased_coin, (5 as usize, 0.7)); 
        println!("my flip coin trace: {:?}", trace);

        #[r_gen] 
        fn flip_my_biased_coin2((n, p) : (usize, f64)) {
            let mut flips = vec![Value::Integer(0); n]; 
            for i in 0..n {
                sample!(flips[i] ~ Distribution::Bernoulli(p)); 
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
                sample!(tests[i] ~ Distribution::Bernoulli(p));
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
                sample!(tests[i] ~ Distribution::Binomial(n, p));
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
                sample!(tests[i] ~ Distribution::Normal(m, s));
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
            sample!(p ~ Distribution::Beta(1.0, 1.0)); 
            sample!(num_heads ~ Distribution::Binomial(100, p.into()));
        }

        //Run the model once in the forward direction and record the observations. 
        let (mut t, _) : (Trace, _)= simulate(&mut my_model, ());
        let mut choices = HashMap::new();
        choices.insert(String::from("num_heads"), *t.get_from_choices_str("num_heads").unwrap()); 

        //Perform importance resampling to get an estimate for the value of p. 
        let mut traces = Vec::new();
        for _ in 0..1000 {
            let (gt, _) : (Trace, _)= generate(&mut my_model, (), &choices);
            traces.push(gt); 
        }
        
        println!("Actuial value for p:\t {:?}", t.get_from_choices_str("p").unwrap()); 
        println!("Generated value for p:\t {:?}", Trace::sample_weighted_traces(&traces).unwrap().get_from_choices_str("p").unwrap());
    }

}

#[allow(dead_code)]
pub mod r_gen {
    use std::rc::Rc;
    use std::collections::HashMap; 
    use std::fmt; 
    use probability::{distribution::Sample, source::Source};
    use rand::{FromEntropy, rngs::{StdRng, ThreadRng}};

    //A value for the probability distribution.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Value {
        Boolean(bool), 
        Integer(i64), 
        Real(f64)
    }

    impl fmt::Display for Value {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Value::Boolean(b) => {
                    write!(formatter, "{}", b)
                }, 
                Value::Integer(i) => {
                    write!(formatter, "{}", i)
                }, 
                Value::Real(r) => {
                    write!(formatter, "{}", r)
                },
            }
            // write!(formatter, "Foo")
        }
    }

    //Implement type conversions
    impl Into<f64> for Value {
        fn into(self) -> f64 { 
            match self {
                Value::Real(v) => v, 
                _ => panic!("Tried to convert non Real value into a f64.")
            }
        }
    }
    impl Into<Value> for f64 {
        fn into(self) -> Value { 
            Value::Real(self)
        }
    }

    impl Into<i64> for Value {
        fn into(self) -> i64 { 
            match self {
                Value::Integer(v) => v, 
                _ => panic!("Tried to convert non Integer value into an i64.")
            }
        }
    }
    impl Into<Value> for i64 {
        fn into(self) -> Value { 
            Value::Integer(self)
        }
    }

    impl Into<bool> for Value {
        fn into(self) -> bool { 
            match self {
                Value::Boolean(v) => v, 
                _ => panic!("Tried to convert non Boolean value into a bool.")
            }
        }
    }
    impl Into<Value> for bool {
        fn into(self) -> Value { 
            Value::Boolean(self)
        }
    }

    //A macro that will handle the distributions for the library. 
    pub mod distributions {
        use probability::prelude::*;
        use rand::{self, FromEntropy, prelude::ThreadRng, rngs::StdRng}; 
        use rand::distributions::Distribution as Distr;
        use super::Value; 
        
        
        //The distributions we can use. 
        #[derive(Debug)]
        pub enum Distribution {
            Bernoulli(f64), 
            Binomial(i64, f64), 
            Normal(f64, f64), 
            Gamma(f64, f64), 
            Beta(f64, f64)
        }
        pub struct Source<T>(pub T);

        impl<T: rand::RngCore> source::Source for Source<T> {
            fn read_u64(&mut self) -> u64 {
                self.0.next_u64()
            }
        }

        impl Distribution {
            //Sample from the distribution. 
            pub fn sample(&self) -> Value {
                match self {
                    Distribution::Bernoulli(p) => {
                        let d = rand::distributions::Bernoulli::new(*p);
                        let v = d.sample(&mut ThreadRng::default());
                        Value::Boolean(v)
                    }, 
                    Distribution::Binomial(n, p) => {
                        let b = probability::distribution::Binomial::new(*n as usize, *p); 
                        Value::Integer(b.sample(&mut Source(StdRng::from_entropy())) as i64)
                    }, 
                    Distribution::Normal(mu, sigma_squared) => {
                        let n = probability::distribution::Gaussian::new(*mu, *sigma_squared); 
                        Value::Real(n.sample(&mut Source(StdRng::from_entropy())))
                    }, 
                    Distribution::Gamma(alpha, beta) => {
                        let g = probability::distribution::Gamma::new(*alpha, *beta); 
                        Value::Real(g.sample(&mut Source(StdRng::from_entropy())))
                    }, 
                    Distribution::Beta(alpha, beta) => {
                        let b = probability::distribution::Beta::new(*alpha, *beta, 0.0, 1.0);
                        Value::Real(b.sample(&mut Source(StdRng::from_entropy()))) 
                    }
                }
            }

            //Compute the liklihood of a value given a distribution. 
            pub fn liklihood(&self, value : &Value) -> Result<f64, &str> {
                match self {
                    Distribution::Bernoulli(p) => {
                        match value {
                            Value::Boolean(b) => {
                                match b {
                                    true  => Ok(p.ln()), 
                                    false => Ok((1.0 - p).ln()), 
                                }
                            }, 
                            _ => Err("Value of wrong type, expected Boolean.")
                        }
                    }, 
                    Distribution::Binomial(n, p) => {
                        match value {
                            Value::Integer(k) => {
                                let norm = probability::distribution::Binomial::new(*n as usize, *p);
                                Ok(norm.mass(*k as usize).ln())
                            }, 
                            _ => Err("Value of wrong type, expected Integer.")
                        }
                    }, 
                    Distribution::Normal(mu, sigma_squared) => {
                        match value {
                            Value::Real(n) => {
                                let norm = probability::distribution::Gaussian::new(*mu, *sigma_squared); 
                                Ok(norm.density(*n).ln())
                            }, 
                            _ => Err("Value of wrong type, expected Real.")
                        }
                    },
                    Distribution::Gamma(alpha, beta) => {
                        match value {
                            Value::Real(n) => {
                                let g = probability::distribution::Gaussian::new(*alpha, *beta); 
                                Ok(g.density(*n).ln())
                            }, 
                            _ => Err("Value of wrong type, expected Real.")
                        }
                    }, 
                    Distribution::Beta(alpha, beta) => {
                        match value {
                            Value::Real(n) => {
                                let b = probability::distribution::Beta::new(*alpha, *beta, 0.0, 1.0);
                                Ok(b.density(*n).ln())
                            }, 
                            _ => Err("Value of wrong type, expected Real.")
                        }
                    }
                }
            }


        }
    }


    //The trace struct. 
    #[derive(Debug, Clone)]
    pub struct Trace {
        log_score : f64, 
        choices : HashMap<String, Value>
    }


    impl Trace {
        //Create a new blank trace. 
        pub fn new() -> Trace {
            Trace{ log_score : 0.0, choices : HashMap::new() }
        }

        //Update the logscore of a trace by adding the given value.  
        fn update_logscore(&mut self, new_value : f64) {
            self.log_score = self.log_score + new_value; 
        }

        //Add a choice to the choicemap. 
        fn add_to_choices(&mut self, name : String, value : Value) {
            // self.choices.insert(&(name.clone()), value);
            self.choices.insert(name, value); 
        }

        //Get a value from the choicemap. 
        pub fn get_from_choices(&mut self, name : &String) -> Option<&Value> {
            self.choices.get(name)
        }
        pub fn get_from_choices_str(&mut self, name : &str) -> Option<&Value> {
            self.choices.get(&String::from(name))
        }

        //A function to return the trace as a string. 
        pub fn get_trace_string(&self) -> String {
            let mut s = String::new(); 
            for (key, value) in &self.choices {
                s.push_str(&format!("{} => {}", key, value));
            }
            s
        }

        //Sample from a vec renormalized by the weight of traces. 
        pub fn sample_weighted_traces(traces : &Vec<Trace>) -> Option<Trace> {
            if traces.len() == 0 {
                None
            } else {
                let values : Vec<f64> = traces.iter().map(|x| x.log_score.exp()).collect();
                let sum : f64 = values.iter().map(|x| x).sum(); 
                let normalized_values : Vec<f64> = values.iter().map(|x| x / sum).collect(); 
                let categorical = probability::distribution::Categorical::new(&normalized_values[..]); 
                
                Some(traces[categorical.sample(&mut distributions::Source(StdRng::from_entropy()))].clone())
            }
        }
    }

    //Implement equivelance for traces based on the log_score. 
    impl PartialEq for Trace {
        fn eq(&self, other: &Trace) -> bool { 
            self.log_score == other.log_score
         }
    }
    impl PartialOrd for Trace {
        fn partial_cmp(&self, other: &Trace) -> std::option::Option<std::cmp::Ordering> { 
            if self.log_score > other.log_score {
                Some(std::cmp::Ordering::Greater)
            } else if self.log_score < other.log_score {
                Some(std::cmp::Ordering::Less)
            } else {
                Some(std::cmp::Ordering::Equal)
            }
        }
    }

    //The function that we will use to generate a trace from a function.
    use distributions::Distribution; 
    pub(crate) fn simulate<F, A, R>(generative_function : &mut F, arguments : A) -> (Trace, R) 
    where 
    F : FnMut(Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, &mut Trace, A) -> R, 
    {
        let sample = |name : &String, dist : Distribution, trace : &mut Trace| {
            let value = dist.sample();                      //Sample a value. 
            let prob = dist.liklihood(&value).unwrap();     //Compute the probability of this value. 
            trace.update_logscore(prob);                    //Update the log score with the pdf. 
            trace.add_to_choices(name.clone(), value);      //Add the choice to the hashmap.
            value 
        }; 
        let mut trace = Trace::new(); 
        let return_value = generative_function(Rc::new(sample), &mut trace, arguments); 
        (trace, return_value)
    }

    //The function that we will use to generate a trace with importance sampling.
    pub(crate) fn generate<F, A, R>(generative_function : &mut F, arguments : A, conditions : &HashMap<String, Value>) -> (Trace, R) 
    where 
    F : FnMut(Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, &mut Trace, A) -> R, 
    {   
        let sample = |name : &String, dist : Distribution, trace : &mut Trace| {
            let mut _value = Value::Real(0.0); 
            _value = if trace.choices.contains_key(name) {
                trace.choices.get(name).unwrap().clone()
            } else {
                dist.sample()
            };
            let prob = dist.liklihood(&_value).unwrap();    //Compute the probability of this value. 
            trace.update_logscore(prob);                    //Update the log score with the pdf. 
            trace.add_to_choices(name.clone(), _value);     //Add the choice to the hashmap.
            _value 
        };
        let mut trace = Trace::new(); 
        for k in conditions.keys() {
            trace.add_to_choices(k.clone(), *conditions.get(k).unwrap()); 
        }
        let return_value = generative_function(Rc::new(sample), &mut trace, arguments);
        (trace, return_value)
    }
    
}