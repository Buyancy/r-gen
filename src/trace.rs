//! Trace objects that represent a run of a generative model.
use std::{collections::HashMap, ops::Index};

use probability::distribution::Sample;
use rand::{FromEntropy, rngs::StdRng};

use super::distributions::{self, Value};

/// A struct to hold all of the random choices made during the execution of a generative model. 
#[derive(Clone, Debug)]
pub struct Choicemap{
    values : HashMap<String, Value>
}

//Implement standard functions for choice maps. 
impl Choicemap {
    /// Create a new, blank choice map. 
    /// # Example
    /// ```
    /// let mut choicemap = Choicemap::new(); 
    /// choicemap.add_choice("p", Value::Real(0.5)); 
    /// ```
    pub fn new() -> Choicemap {
        Choicemap{ values : HashMap::new() }
    }

    /// Create a new choicemap with given choices in it. 
    /// # Example 
    /// ```
    /// let mut choicemap = Choicemap::from(vec![("p", Value::Real(0.5))]); 
    /// ```
    pub fn from(choices : Vec<(&str, Value)>) -> Choicemap {
        let mut res = Choicemap::new(); 
        choices.iter().for_each(|(s, v)| res.add_choice(*s, *v)); 
        res
    }

    /// Add a choice to this choicemap. 
    /// # Example
    /// ```
    /// let mut choicemap = Choicemap::new(); 
    /// choicemap.add_choice("p", Value::Real(0.5)); 
    /// ```
    pub fn add_choice(&mut self, identifier : &str, value : Value) {
        self.values.insert(identifier.to_string(), value); 
    }

    /// Get a list of the choices that were made in the generative model. 
    pub fn get_choices(&self) -> Vec<(&str, Value)> {
        self.values.keys().map(|k| (k.as_str(), self.values.get(k).unwrap().clone())).collect() 
    }

    /// Check whether or not the given key is already in the choicemap. 
    pub fn contains_key(&self, key : &str) -> bool {
        self.values.contains_key(key)
    }
}

impl Index<&str> for Choicemap {
    type Output = Value;

    fn index(&self, index: &str) -> &Self::Output {
        match self.values.get(index) {
            Some(v) => v, 
            None => panic!("Value not present in choicemap.")
        }
    }
}

impl Index<&String> for Choicemap {
    type Output = Value;

    fn index(&self, index: &String) -> &Self::Output {
        match self.values.get(index.as_str()) {
            Some(v) => v, 
            None => panic!("Value not present in choicemap.")
        }
    }
}

/**
The trace struct. This holds information about the execution of a gnerative model. 
*/
#[derive(Debug, Clone)]
pub struct Trace {
    /// The log joint liklihood of all of the random decisions in the trace. 
    pub log_score : f64, 
    /// The Choicemap that holds the list of the actuial decisions that were made in the execution of the generative model.
    pub choices : Choicemap
}


impl Trace {
    /**
    Create a new blank trace. It begins with an empty choice map and a log score of 0 (which corresponds to a 
    probability of 1.0 when exponentiated.)
    */
    pub fn new() -> Trace {
        Trace{ log_score : 0.0, choices : Choicemap::new() }
    }

    /**
    Update the logscore of a given trace. 
    */
    pub(crate) fn update_logscore(&mut self, new_value : f64) {
        self.log_score = self.log_score + new_value; 
    }

    /**
    Return a string that discribes the random decisions made by the model in this trace.
    # Example 
    ```
    #[r_gen]
    fn my_biased_coin_model(():()){
        sample!(p ~ Distribution::Beta(1.0, 1.0));                    //Sample p from a uniform. 
        sample!(num_heads ~ Distribution::Binomial(100, p.into()));   //Flip 100 coins where P(Heads)=p
    }
    let (trace, result) = generate(&mut my_biased_coin_model, ()); 
    println!("{}", trace.get_trace_string()); 
    ```
    */
    pub fn get_trace_string(&self) -> String {
        let mut s = String::new(); 
        for (key, value) in &self.choices.get_choices() {
            s.push_str(&format!("{} => {}\n", key, value));
        }
        s
    }

    /**
    Sample a trace from a vector of traces according to a categorical distribution. The weights for the distribution are 
    the scores of the traces rescaled by a normalizing constant. This function is intended to be used in an importance
    resampling algorithm.
    */
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
