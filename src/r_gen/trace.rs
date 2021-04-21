//A module that will handle traces for our library.
use std::{collections::HashMap, ops::Index};

use probability::distribution::Sample;
use rand::{FromEntropy, rngs::StdRng};

use super::distributions::{self, Value};

//A choicemap struct that will be a part of the trace. 
#[derive(Clone, Debug)]
pub struct Choicemap{
    values : HashMap<String, Value>
}

//Implement standard functions for choice maps. 
impl Choicemap {
    //Create a new, blank choice map. 
    pub fn new() -> Choicemap {
        Choicemap{ values : HashMap::new() }
    }

    //Create a new choicemap with known choices in it. 
    pub fn from(choices : Vec<(&str, Value)>) -> Choicemap {
        let mut res = Choicemap::new(); 
        choices.iter().for_each(|(s, v)| res.add_choice(*s, *v)); 
        res
    }

    //Add a choice to this choicemap. 
    pub fn add_choice(&mut self, identifier : &str, value : Value) {
        self.values.insert(identifier.to_string(), value); 
    }

    //Get a list of the choices that we made. 
    pub fn get_choices(&self) -> Vec<(&str, Value)> {
        self.values.keys().map(|k| (k.as_str(), self.values.get(k).unwrap().clone())).collect() 
    }

    //Return whether or not we have a given key in it. 
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

//The trace struct. 
#[derive(Debug, Clone)]
pub struct Trace {
    pub log_score : f64, 
    pub choices : Choicemap
}


impl Trace {
    //Create a new blank trace. 
    pub fn new() -> Trace {
        Trace{ log_score : 0.0, choices : Choicemap::new() }
    }

    //Update the logscore of a trace by adding the given value.  
    pub fn update_logscore(&mut self, new_value : f64) {
        self.log_score = self.log_score + new_value; 
    }

    //A function to return the trace as a string. 
    pub fn get_trace_string(&self) -> String {
        let mut s = String::new(); 
        for (key, value) in &self.choices.get_choices() {
            s.push_str(&format!("{} => {}\n", key, value));
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
