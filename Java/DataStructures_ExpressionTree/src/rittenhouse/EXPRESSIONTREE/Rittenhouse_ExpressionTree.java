package rittenhouse.EXPRESSIONTREE;

import java.util.Scanner;

import rittenhouse.POSTFIXCALC.LinkedStack;
import rittenhouse.POSTFIXCALC.Converter;

// Simone Rittenhouse Homework 3
public class Rittenhouse_ExpressionTree {
	
	// inner Node class
	public static class Node{
		Object element;
		Node left;
		Node right;
		
		public Node (Object o) {
			this(o, null, null);
		}
		
		public Node(Object o, Node left, Node right) {
			this.element = o;
			this.left = left;
			this.right = right;
		}
		
		public String toString() {
			return element.toString();
		}
	}
	
	// instance variables
	private Node root;
	
	public Rittenhouse_ExpressionTree() {
		root = null;
	}
	
	// methods for setting children
	
	public void setRoot(Object element) {
		this.root = new Node(element);
	}
	
	public Node getRoot() {
		return this.root;
	}
	
	public void setLeftChild(Node t, Node l) {
		t.left = l;
	}
	
	public void setRightChild(Node t, Node r) {
		t.right = r;
	}
	
	// HOMEWORK REQUIRED METHODS
	
	public static Rittenhouse_ExpressionTree convert(String postfix) {
		// tokenizing postfix expression
		String[] tokens = postfix.split(" ");
		
		// creating empty stack (from textbook)
		LinkedStack<Rittenhouse_ExpressionTree> stack = new LinkedStack<>();
		
		for(String token : tokens) {
			if(Character.isDigit(token.charAt(0))) { // for numbers
				Rittenhouse_ExpressionTree number = new Rittenhouse_ExpressionTree();
				number.setRoot(token);
				stack.push(number);
			}else if(token.equals("*") || token.equals("/") || // for operators
					 token.equals("+") || token.equals("^") || 
					 token.equals("-")) {
				Rittenhouse_ExpressionTree operator = new Rittenhouse_ExpressionTree();
				operator.setRoot(token);
				
				// popping two tokens off the stack
				Rittenhouse_ExpressionTree right = stack.pop();
				Rittenhouse_ExpressionTree left = stack.pop();
				operator.setLeftChild(operator.getRoot(), left.getRoot());
				operator.setRightChild(operator.getRoot(), right.getRoot());
				
				// pushing new tree onto the stack
				stack.push(operator);
			}
		}
		// returning finished tree
		Rittenhouse_ExpressionTree tree = stack.pop();
		return tree;
	}
	
	public void prefix() {
		prefix(root);
		System.out.println();
	}
	
	private void prefix(Node t) {
		if(t != null) {
			System.out.print(t);
			prefix(t.left);
			prefix(t.right);
		}
	}
	
	public void infix() {
		infix(root);
		System.out.println();
	}
	
	private void infix(Node t) {
		if(t != null) {
			if(t.left != null) {
				System.out.print("("); // printing parentheses and avoiding leaves
			}
			infix(t.left);
			System.out.print(t);
			infix(t.right);
			
			if(t.right != null) {
				System.out.print(")"); // printing parentheses and avoiding leaves
			}
		}
	}
	
	public void postfix() {
		postfix(root);
		System.out.println();
	}
	
	private void postfix(Node t) {
		if(t != null) {
			postfix(t.left);
			postfix(t.right);
			System.out.print(t);
		}
	}

	public static void main(String[] args) {
		
		Scanner scan = new Scanner(System.in);
		Converter calc = new Converter(""); // from Homework 2 PostFix Calculator
		
		while(true) { // will keep prompting until user terminates program (as said in instructions)
			
			System.out.println("Please enter an infix expression: ");
			String testing = scan.nextLine();
			
			// going from input to postfix using Homework 2
			calc.setInfix(testing);
			String postfix = calc.toPostFix();
			
			// going from postfix to Expression Tree
			Rittenhouse_ExpressionTree tree = convert(postfix);
			
			// tree traversals
			System.out.print("Prefix: ");
			tree.prefix();
			System.out.print("Infix: ");
			tree.infix();
			System.out.print("Postfix: ");
			tree.postfix();
			
			System.out.println();
			
		}

	}

}
