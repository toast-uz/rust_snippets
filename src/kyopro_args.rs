#![allow(dead_code)]

use regex::Regex;
use itertools::Itertools;

pub struct Args {
    args_str: String,
}

impl Args {
    pub fn new() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let args_str = args[1..].iter().join(" ");
        Self { args_str }
    }

    pub fn get<T: std::str::FromStr>(&self, arg_name: &str) -> Option<T> {
        let re_str = format!(r"-{}=([\d.-]+)", arg_name);
        let re = Regex::new(&re_str).unwrap();
        let Some(captures) = re.captures(&self.args_str) else { return None; };
        captures[1].parse().ok()
    }
}

pub mod os_env {
    const PREFIX: &str = "AHC_PARAMS_";

    pub fn get<T: std::str::FromStr>(name: &str) -> Option<T> {
        let name = format!("{}{}", PREFIX, name.to_uppercase());
        std::env::var(name).ok()?.parse().ok()
    }
    pub fn get_wo_prefix<T: std::str::FromStr>(name: &str) -> Option<T> {
        std::env::var(name).ok()?.parse().ok()
    }
    pub fn atcoder() -> bool {
        get_wo_prefix::<usize>("atcoder").unwrap_or(0) == 1
    }
}

fn main() {
    // コマンドライン引数の取得
    let args = Args::new();
    let x: usize = args.get::<usize>("x").unwrap();
    let y: f64 = args.get::<f64>("y").unwrap();
    println!("x: {}, y: {}", x, y);

    // 環境変数の取得
    let x: usize = os_env::get::<usize>("x").unwrap();
    let y: f64 = os_env::get::<f64>("y").unwrap();
    println!("x: {}, y: {}", x, y);
}
