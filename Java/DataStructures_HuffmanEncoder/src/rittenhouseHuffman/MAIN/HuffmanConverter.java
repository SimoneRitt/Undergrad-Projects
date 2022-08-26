package rittenhouseHuffman.MAIN;

import java.io.BufferedReader;
import java.io.FileReader;

/**
 * @author simonerittenhouse
 *
 */
/**
 * @author simonerittenhouse
 *
 */
public class HuffmanConverter {
	
	public static final int NUMBER_OF_CHARACTERS = 256;
	
	private String contents;
	private HuffmanTree huffmanTree;
	private int count[];
	private String code[];
	private int uniqueChars = 0;
	
	// constructor
	
	public HuffmanConverter(String input) {
		this.contents = input;
		this.count = new int[NUMBER_OF_CHARACTERS];
		this.code = new String[NUMBER_OF_CHARACTERS];
	}

	// methods
	
	public void recordFrequencies() {
		for(int x = 0; x < this.contents.length(); x ++) {
			// if new unique character
			if(this.count[(int)this.contents.charAt(x)] == 0) {
				this.uniqueChars += 1;
			}
			// increasing the count
			this.count[(int)this.contents.charAt(x)] += 1;
		}
	}
	
	public void frequenciesToTree() {
		
		HuffmanNode[] nodes = new HuffmanNode[this.uniqueChars];
		int index = 0;
		for(int x = 0; x < this.count.length; x += 1) {
			if(this.count[x] != 0) {
				String letter = "" + (char)x;
				Double frequency = Double.valueOf(this.count[x]);
				nodes[index] = new HuffmanNode(letter, frequency);
				index ++;
			}
		}
		
		// creating heap from node array
		BinaryHeap ans = new BinaryHeap(nodes);
		
		// printing the heap
		ans.printHeap();
		
		// creating tree from heap
		this.huffmanTree = HuffmanTree.createFromHeap(ans);
	}
	
	public void treeToCode() {
		// if empty tree, return
		if(this.huffmanTree.root == null) {
			return;
		}
		// else, fill code[]
		treeToCode(this.huffmanTree.root, "");
	}
	
	private void treeToCode(HuffmanNode t, String s) {
		// NOTE: I reversed the encoding for left and right
		// to match the sample output provided
		if(t.letter.length() > 1) {
			treeToCode(t.left, s + "1");
			treeToCode(t.right, s + "0");
		}else {
			// fill code[]
			this.code[(int)t.letter.charAt(0)] = s;
			
			// printing tree
			System.out.println(t.letter + "=" + s);
		}
	}
	
	public String encodeMessage() {
		String encodedMessage = "";
		
		for(int x = 0; x < this.contents.length(); x ++) {
			encodedMessage += this.code[(int)this.contents.charAt(x)];
		}
		
		return encodedMessage;
	}
	
	public static String readContents(String filename) {
		String content = "";
		
		try {
			BufferedReader BR = new BufferedReader(new FileReader(filename));
			
			String line = "";
			while((line = BR.readLine()) != null) {
				for(int x = 0; x < line.length(); x ++) {
					content += line.charAt(x);
				}
				content += "\n";
			}
			
			BR.close();
			
		}catch(Exception e) {
			System.out.println(e);
		}
		
		return content;
	}
	
	public String decodeMessage(String encodeStr) {
		String decodedMessage = "";
		
		String code = "";
		for(int x = 0; x < encodeStr.length(); x ++) {
			code += encodeStr.charAt(x);
			for(int y = 0; y < this.code.length; y ++) {
				if (code.equals(this.code[y])){
					decodedMessage += (char)y;
					code = "";
				}
			}
		}
		
		return decodedMessage;
	}
	
	/*
	 * public static void main(String[] args) { // getting contents from filename
	 * String input = readContents(args[0]);
	 * 
	 * // initializing converter HuffmanConverter hc = new HuffmanConverter(input);
	 * 
	 * // recording char frequencies hc.recordFrequencies();
	 * 
	 * // creating Huffman Tree hc.frequenciesToTree();
	 * 
	 * // getting Huffman encodings System.out.println(); hc.treeToCode();
	 * 
	 * // encoding message System.out.println(); String edcdMess =
	 * hc.encodeMessage(); System.out.println("Huffman Encoding:");
	 * System.out.println(edcdMess);
	 * 
	 * // getting message sizes int ASCIISize = 8 * input.length(); int HuffmanSize
	 * = edcdMess.length();
	 * 
	 * System.out.println(); System.out.println("Message size in ASCII encoding: " +
	 * ASCIISize); System.out.println("Message size in Huffman coding: " +
	 * HuffmanSize);
	 * 
	 * // decoding message System.out.println();
	 * System.out.println(hc.decodeMessage(edcdMess));
	 * 
	 * }
	 */

}
