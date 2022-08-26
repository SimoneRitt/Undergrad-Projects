package homework.main;

import java.util.Random;

public class RittenhouseHomework2 {
	
	// Creating display matrix method to show output
	
	public static void displayMatrix(int[][] a) {
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a[x].length; y ++) {
				System.out.print(String.format("%-4d", a[x][y]));
			}
			
			System.out.println();
		}
	}
	
	public static void displayMatrix(double[][] a) {
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a[x].length; y ++) {
				
				System.out.printf("%.2f", a[x][y]);
				System.out.print("  ");
			}
			
			System.out.println();
		}
	}
	
	// TASK ONE START
	
	public static int[][] constrainedMatrix(){
		int[][] A = new int[5][5];
		
		for(int x = 0; x < A.length; x ++) {
			for(int y = 0; y < A.length; y ++) {
				
				if(x == 0) { // row 1
					A[x][y] = 1;
				}
				
				else if(x == 1 || x == 2) { // rows 2 and 3
					Random r = new Random();
					
					int num = r.nextInt(10);
					
					A[x][y] = num;
				}
				
				else if(x == 3) { // row 4
					A[x][y] = A[1][y] + A[2][y];
				}
				
				else { // row 5
					A[x][y] = (int)Math.pow((A[1][y] - A[2][y]), 2);
				}
			}
		}
		
		return A;
	}
	
	// TASK ONE END
	
	// TASK TWO START
	
	public static int[][] matrixModification(int[][] a){
		
		int[][] modifiedMatrix = new int[a.length][a[0].length];
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a[x].length; y ++) {
				if((x == 0 || x == 3) & (a[x][y] % 2 != 0)) { // rows 1 and 4
					modifiedMatrix[x][y] = a[x][y] * 3;
				}
				else if((x == 1 || x == 2) & (a[x][y] % 2 == 0)) { // rows 2 and 3
					modifiedMatrix[x][y] = a[x][y] * 2;
				}
				else { // all other rows
					modifiedMatrix[x][y] = a[x][y];
				}
			}
		}
		
		return modifiedMatrix;
	}
	
	// TASK TWO END
	
	// TASK THREE START
	
	public static double[] minMax(int[] a) {
		double[] minMax = new double[2];
		
		double min = a[0];
		double max = a[0];
		
		for(int x = 0; x < a.length; x ++) {
			if(a[x] > max) {
				max = a[x];
			}
			else if(a[x] < min) {
				min = a[x];
			}
		}
		
		minMax[0] = min;
		minMax[1] = max;
		
		return minMax;
	}
	
	public static double[] meanSum(int[] a) {
		double[] meanSum = new double[2];
		
		double sum = 0;
		
		for(int x = 0; x < a.length; x ++) {
			sum += a[x];
		}

		double mean = sum / a.length;
		
		meanSum[0] = mean;
		meanSum[1] = sum;
		
		return meanSum;
	}
	
	public static double variance(int[] a) {
		double variance = 0;
		
		double mu = meanSum(a)[0];
		double squaredDiff = 0;
		
		for(int x = 0; x < a.length; x ++) {
			squaredDiff += Math.pow((a[x] - mu), 2);
		}
		
		variance = squaredDiff / a.length;
		
		return variance;
	}
	
	public static double[][] statsSummary(int[][] a) {
		double[][] stats = new double[5][5];
		
		for(int x = 0; x < stats.length; x ++) {
			
			stats[x][0] = minMax(a[x])[0];
			stats[x][1] = meanSum(a[x])[0];
			stats[x][2] = minMax(a[x])[1];
			stats[x][3] = meanSum(a[x])[1];
			stats[x][4] = variance(a[x]);
			
		}
		
		return stats;
	}
	
	// TASK THREE END
	
	// TASK FOUR START
	
	public static boolean checkSquare(int[][] a) {
		boolean square = true;
		
		for(int x = 0; x < a.length; x ++) {
			if(a[x].length != a.length) {
				square = false;
			}
		}
		
		return square;
	}
	
	public static boolean consecutive(int[] a) {
		boolean consecutive = true;
		
		for(int x = 0; x < a.length -1; x ++) {
			if(a[x] != a[x+1]) {
				consecutive = false;
			}
		}
		
		return consecutive;
	}
	
	public static boolean checkConsecutiveArray(int[][] a) {
		boolean consecutiveCheck = false;
		
		int[] diagonal1 = new int[4];
		int[] diagonal2 = new int[4];
		
		int diagonal2index = a.length - 1;
		
		for(int x = 0; x < a.length; x ++) {
			
			int[] row = new int[4];
			int[] column = new int[4];
					
			for(int y = 0; y < 4; y ++) {
				row[y] = a[x][y];
				column[y] = a[y][x];
			}
			
			if(consecutive(row) == true || consecutive(column) == true) {
				consecutiveCheck = true;
			}
			
			if(x < 4) {
				diagonal1[x] = a[x][x];
				diagonal2[x] = a[x][diagonal2index];
				
				diagonal2index -= 1;
			}
			
		}
		
		if(consecutive(diagonal1) == true || consecutive(diagonal2) == true) {
			consecutiveCheck = true;
		}
		
		return consecutiveCheck;
	}
	
	public static boolean check4(int[][] a) {
		boolean check = true;
		
		if(checkSquare(a) == false) {
			check = false;
		}
		
		else if(a.length != 6) {
			check = false;
		}
		
		else if(checkConsecutiveArray(a) == false) {
			check = false;
		}
		
		return check;
	}
	
	// TASK FOUR END
	
	// TASK FIVE START
	
	public static int[] countEvensOdds(int[][] a) {
		int[] evensOdds = new int[2];
		
		int evens = 0;
		int odds = 0;
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				if(a[x][y] % 2 == 0) {
					evens += 1;
				}
				
				else {
					odds += 1;
				}
			}
		}
		
		evensOdds[0] = evens;
		evensOdds[1] = odds;
		
		return evensOdds;
	}
	
	public static int[] buildEvenArray(int[][] a) {
		int[] evenArray = new int[countEvensOdds(a)[0]];
		
		int index = 0;
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				if(a[x][y] % 2 == 0) {
					evenArray[index] = a[x][y];
					index += 1;
				}
			}
		}
		
		return evenArray;
	}
	
	public static int[] buildOddArray(int[][] a) {
		int[] oddArray = new int[countEvensOdds(a)[1]];
		
		int index = 0;
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				if(a[x][y] % 2 != 0) {
					oddArray[index] = a[x][y];
					index += 1;
				}
			}
		}
		
		return oddArray;
	}
	
	public static int[][] evenFirst(int[][] a) {
		int[][] orderedArray = new int[a.length][a.length];
		
		int[] evenArray = buildEvenArray(a);
		int[] oddArray = buildOddArray(a);
		
		int evenIndex = 0;
		int oddIndex = 0;
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				if(evenIndex < evenArray.length) {
					orderedArray[x][y] = evenArray[evenIndex];
					evenIndex += 1;
				}
				
				else {
					orderedArray[x][y] = oddArray[oddIndex];
					oddIndex += 1;
				}
			}
		}
		
		return orderedArray;
	}
	
	// TASK FIVE END

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		// Header
		
		System.out.println("Simone Rittenhouse: Homework 2");
				
		// Task 1 Output
		
		System.out.println('\n' + "------------------");	
		System.out.println("Task 1 Output:" + '\n');
		
		displayMatrix(constrainedMatrix());
		
		// Task 2 Output
		
		System.out.println('\n' + "------------------");
		System.out.println("Task 2 Output:" + '\n');
		
		int[][] A = {{1, 3, 2, 7, 4}, {4, 7, 5, 3, 9}, {3, 3, 6, 8, 2}, {1, 5, 7, 2, 2}, {3, 9, 9, 2, 3}};
		
		System.out.println("Unmodified Matrix:");
		displayMatrix(A);
		
		System.out.println('\n' + "Modified Matrix:");
		displayMatrix(matrixModification(A));
		
		// Task 3 Output
		
		System.out.println('\n' + "------------------");
		System.out.println("Task 3 Output:" + '\n');
		
		System.out.println("Original Matrix:");
		displayMatrix(A);
		
		System.out.println('\n' + "Statistical Summary:");
		displayMatrix(statsSummary(A));
		
		// Task 4 Output
		
		System.out.println('\n' + "------------------");
		System.out.println("Task 4 Output:" + '\n');
		
		int[][] R = {{1, 1, 1, 1, 4, 3}, {4, 7, 5, 3, 9, 4}, {3, 3, 6, 8, 2, 5}, {1, 5, 7, 2, 2, 6}, {3, 9, 9, 2, 3, 7}, {6, 3, 2, 1, 0, 8}};
		int[][] D1 = {{1, 1, 6, 1, 4, 3}, {4, 1, 5, 3, 9, 4}, {3, 3, 1, 8, 2, 5}, {1, 5, 7, 1, 2, 6}, {3, 9, 9, 2, 3, 7}, {6, 3, 2, 1, 0, 8}};
		int[][] C = {{1, 1, 6, 1, 4, 3}, {1, 1, 5, 3, 9, 4}, {1, 3, 2, 8, 2, 5}, {1, 5, 7, 5, 2, 6}, {3, 9, 9, 2, 3, 7}, {6, 3, 2, 1, 0, 8}};
		int[][] D2 = {{1, 1, 6, 1, 4, 1}, {4, 1, 5, 3, 1, 4}, {3, 3, 3, 1, 2, 5}, {1, 5, 1, 1, 2, 6}, {3, 2, 9, 2, 3, 7}, {5, 3, 2, 1, 0, 8}};
		int[][] F1 = {{1, 1, 1, 1}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
		int[][] F2 = {{1, 2, 3, 4}, {5, 5, 5, 5}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}; 
		int[][] F3 = {{1, 2, 1, 5, 4, 3}, {4, 7, 5, 3, 9, 4}, {3, 3, 6, 8, 2, 5}, {1, 5, 7, 2, 2, 6}, {3, 9, 9, 2, 3, 7}, {6, 3, 2, 1, 0, 8}};
		
		System.out.println("First 4 Row Values the Same:");
		displayMatrix(R);
		System.out.println();
		System.out.println(check4(R));
		
		System.out.println('\n' + "First 4 Main Diagonal Values the Same:");
		displayMatrix(D1);
		System.out.println();
		System.out.println(check4(D1));
		
		System.out.println('\n' + "First 4 Column Values the Same:");
		displayMatrix(C);
		System.out.println();
		System.out.println(check4(C));
		
		System.out.println('\n' + "First 4 Main Diagonal Values the Same:");
		displayMatrix(D2);
		System.out.println();
		System.out.println(check4(D2));
		
		System.out.println('\n' + "Not of Length 6:");
		displayMatrix(F1);
		System.out.println();
		System.out.println(check4(F1));
		
		System.out.println('\n' + "Not a Square Matrix:");
		displayMatrix(F2);
		System.out.println();
		System.out.println(check4(F2));
		
		System.out.println('\n' + "None of the First Four Values are the Same:");
		displayMatrix(F3);
		System.out.println();
		System.out.println(check4(F3));
		
		// Task 5 Output
		
		System.out.println('\n' + "------------------");
		System.out.println("Task 5 Output:" + '\n');
		
		System.out.println("Unordered Matrix:");
		displayMatrix(A);
		
		System.out.println('\n' + "Ordered Matrix:");
		displayMatrix(evenFirst(A));
		
	}

}
