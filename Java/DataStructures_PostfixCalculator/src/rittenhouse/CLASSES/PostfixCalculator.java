package rittenhouse.CLASSES;

import java.util.Scanner;

import rittenhouse.STACK.LinkedStack;

public class PostfixCalculator {
	
	// instance variable
	private Converter C;
	
	// getter and setter
	public Converter getC() {
		return C;
	}

	public void setC(Converter c) {
		C = c;
	}
	
	// so you can change the infix from Calculator
	public void setInfix(String infix) {
		this.C.setInfix(infix);
	}
	
	// constructor
	public PostfixCalculator(String infix) {
		this.C = new Converter(infix);
	}
	
	// helper methods to evaluate operator
	private boolean isOperator(char c) {
		if (c == '*' || c == '/' || 
            c == '+' || c == '^' || 
            c == '-') {
			return true;
		}else {
			return false;
		}
	}
	
	// helper method to operate
	private double Operate(double a, double b, char c) {
		if(c == '*') {
			return a * b;
		}else if(c == '/') {
			return a/b;
		}else if(c == '+') {
			return a + b;
		}else if(c == '-') {
			return a-b;
		}else if(c == '^') {
			return Math.pow(a, b);
		}else {
			return 0;
		}
	}
	
	// method for evaluating postfix
	private double Calculation(String postfix) {
		double evaluated = 0;
		
		LinkedStack<String> stack = new LinkedStack<>();
		
		String num = "";
		for(int x = 0; x < postfix.length(); x ++) {
			char c = postfix.charAt(x);
			// creating numbers to push onto stack
			if(Character.isDigit(c)) {
				num += c;
			}
			// pushing numbers
			else if(c == ' ' & Character.isDigit(postfix.charAt(x-1))) {
				stack.push(num);
				num = "";
			}
			else if(isOperator(c)) {
				// popping two operands
				double sec = Double.parseDouble(stack.pop());
				double first = Double.parseDouble(stack.pop());
				
				// evaluating them
				double ans = Operate(first, sec, c);
				
				// pushing answer back to stack
				stack.push(String.valueOf(ans));
				
			}
		}
		
		if(!stack.isEmpty()) {
			evaluated = Double.parseDouble(stack.pop());
		}
		
		return evaluated;
	}
	
	// method for creating output
	public void calculate() {
		
		// getting postfix expression and value
		String postfix = C.toPostFix();
		double value = Calculation(postfix);
		
		// generating output
		System.out.println("converted to postfix: " + postfix);
		System.out.println("answer is " + value);
		
	}
	
	public static void main(String[] args) {
		
		// testing calculator with fixed input
		System.out.println("Commencing Calculator Test: \n");
		System.out.println("infix = (4+8)*(6-5)/((3-2)*(2+2))");
		
		PostfixCalculator myCalc = new PostfixCalculator("(4+8)*(6-5)/((3-2)*(2+2))");
		myCalc.calculate();
		
		Scanner scan = new Scanner(System.in);
		System.out.println("\nWould you like to evaluate an expression? (yes/no) ");
		String ans = scan.next();
		
		if(ans.equals("yes")) {
			// getting infix expression
			System.out.println("Type your infix expression: ");
			scan.nextLine();
			String infix = scan.nextLine();
			
			// calculating postfix and evaluating
			myCalc.setInfix(infix);
			myCalc.calculate();
		}
		System.out.println("\nHave a nice day!");
		
	}
	
}
