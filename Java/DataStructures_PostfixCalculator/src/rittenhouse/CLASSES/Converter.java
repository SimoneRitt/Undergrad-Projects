package rittenhouse.CLASSES;

import java.util.ArrayList;
import java.util.List;

import rittenhouse.STACK.LinkedStack;

public class Converter {
	
	// instance variable
	private String infix;
	
	// getter and setter
	public String getInfix() {
		return infix;
	}

	public void setInfix(String infix) {
		this.infix = infix;
	}
	
	// constructor
	public Converter(String infix) {
		this.infix = infix;
	}
	
	// helper methods
	
	// parse method from homework instructions
	private static List<String> parse(char[] input) {
	    List<String> parsed = new ArrayList<String>();
	    for (int i = 0; i < input.length; ++i) {
	        char c = input[i];
	        if (Character.isDigit(c)) {
	            String number = input[i] + "";
	            for (int j = i + 1; j < input.length; ++j) {
	                if (Character.isDigit(input[j])) {
	                    number += input[j];
	                    i = j;
	                } else {
	                    break;
	                }
	            }
	            parsed.add(number);
	        } else if (c == '*' || c == '/' || 
	                   c == '+' || c == '^' || 
	                   c == '-' || c == '(' || c == ')') {
	            parsed.add(c + "");
	        }
	    }
	    
	    return parsed;
	}
	
	// assigning precedence values to operators for comparison
	private static int precedence(String op) {
		int value;
		if(op == null) {
			return 0;
		}
		switch (op) {
			case "^":
				value = 4;
				break;
			case "*":
				value = 3;
				break;
			case "/":
				value = 3;
				break;
			case "+":
				value = 2;
				break;
			case "-":
				value = 2;
				break;
			case "(":
				value = 1;
				break;
			default:
				value = 0;
				break;
		}
		
		return value;
	}
	
	private static String PostFix(List<String> parsed) {
		String output = "";
		
		// creating Stack
		LinkedStack<String> stack = new LinkedStack<>();
		
		// traversing the input
		for (String s : parsed) {
			// appending operands to output
			if (Character.isDigit(s.charAt(0))){
				output += s + " ";
			}
			// pushing open parentheses
			else if (s.equals("(")){
				stack.push(s);
			}
			// popping closed parentheses
			else if (s.equals(")")) {
				while(!stack.top().equals("(")) {
					output += stack.pop() + " ";
				}
				stack.pop(); // popping "("
			}
			// pushing higher precedence
			else if (precedence(stack.top()) < precedence(s)){
				stack.push(s);
			}
			// popping lower precedence
			else if (precedence(stack.top()) >= precedence(s)){
				while(precedence(s) <= precedence(stack.top())) {
					if(!stack.isEmpty()) {
						output += stack.pop() + " ";
					}
					else {
						break;
					}
				}
				stack.push(s);
			}
			
		}
		
		// popping anything remaining in the stack
		while(!stack.isEmpty()) {
			output += stack.pop() + " ";
		}
		
		return output;
	}
	
	// toPostFix method
	public String toPostFix() {
	
		// getting infix ready to be tokenized
		char[] input = new char[this.infix.length()];
		for(int x = 0; x < this.infix.length(); x ++) {
			input[x] = this.infix.charAt(x);
		}
		// parsing through infix
		List<String> parsed = parse(input);
		
		// creating PostFix expression
		String output = PostFix(parsed);
		
		return output;
	}

}
