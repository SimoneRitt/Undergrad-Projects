package rittenhouseHuffman.MAIN;

public class HuffmanTree {
	
	HuffmanNode root;
	
	public HuffmanTree(HuffmanNode huff) {
		this.root = huff;
	}
	
	public void printLegend() {
		printLegend(root, "");
	}
	
	private void printLegend(HuffmanNode t, String s) {
		if(t == null) { // if HuffmanTree is empty
			return;
		}
		if(t.letter.length() > 1) {
			printLegend(t.left, s + "0");
			printLegend(t.right, s + "1");
		}else {
			System.out.println(t.letter + "=" + s);
		}
	}
	
	public static BinaryHeap legendToHeap(String legend) {
		String[] legendArr = legend.split(" ");
		// making sure there are equal numbers of letters and frequencies
		// making sure legend isn't empty
		if(legendArr.length < 2 | legendArr.length %2 != 0) {
			return new BinaryHeap(); // returning empty heap
		}
		
		// creating HuffmanNode objects and storing them in array
		HuffmanNode[] nodes = new HuffmanNode[legendArr.length /2];
		int index = 0;
		for(int x = 0; x < legendArr.length; x += 2) {
			String letter = legendArr[x];
			Double frequency = Double.parseDouble(legendArr[x+1]);
			nodes[index] = new HuffmanNode(letter, frequency);
			index ++;
		}
		
		// creating heap from node array
		BinaryHeap ans = new BinaryHeap(nodes);
		
		return ans;
	}
	
	public static HuffmanTree createFromHeap(BinaryHeap b) {
		if (b.isEmpty()){
			// return empty tree
			return new HuffmanTree(null);
		}
		while(b.getSize() > 1) {
			HuffmanNode left = (HuffmanNode) b.deleteMin();
			HuffmanNode right = (HuffmanNode) b.deleteMin();
			
			// inserting new node with children as two min items in heap
			b.insert(new HuffmanNode(left, right));
		}
		// returning HuffmanTree with root set as remaining item in heap
		return new HuffmanTree((HuffmanNode) b.deleteMin());
	}
	
	// Main Method (from Part 1)
	
	  public static void main(String[] args) { 
		  String test =  "E 2 V 1 N 1";
		  BinaryHeap bheap = legendToHeap(test); bheap.printHeap();
		  System.out.println();
		  HuffmanTree htree = createFromHeap(bheap); htree.printLegend();
	  
	  }
	 

}
