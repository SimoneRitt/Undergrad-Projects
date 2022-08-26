package rittenhouseHuffman.MAIN;

public class HuffmanNode implements Comparable{
	
	public String letter;
	public Double frequency;
	public HuffmanNode left, right;
	
	public HuffmanNode(String letter, Double frequency) {
		this.letter = letter;
		this.frequency = frequency;
		this.left = null;
		this.right = null;
	}
	
	public HuffmanNode(HuffmanNode left, HuffmanNode right) {
		this.left = left;
		this.right = right;
		this.letter = left.letter + right.letter;
		this.frequency = left.frequency + right.frequency;
	}

	@Override
	public int compareTo(Object o) {
		// making sure o is a HuffmanNode
		if(!(o instanceof HuffmanNode)) {
			 return 0;
		}
		
		// casting and comparing
		HuffmanNode huff = (HuffmanNode) o;
		return this.frequency.compareTo(huff.frequency);
	}
	
	@Override
	public String toString() {
		return "<"+this.letter+", "+this.frequency+">";
	}

}
