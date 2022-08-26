package homework.main;

import java.util.Random;

public class RittenhouseHomework1 {
	
	// Task 1 START
	
	public static void displayMatrix(int[][] a) {
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				System.out.print(a[x][y] + " ");
			}
			System.out.println();
		}
	}
	
	// Task 1 END
	
	// Task 2 START
	
	public static int[][] buildMatrix(int[] a){
		int[][] newMatrix = new int[3][3];
		
		int index = 0;
		
		for(int x = 0; x < newMatrix.length; x ++) {
			for(int y = 0; y < newMatrix.length; y ++) {
				newMatrix[x][y] = a[index];
				index += 1;
			}
		}
		
		return newMatrix;
	}
	
	// Task 2 END
	
	// Task 3 START
	
	public static int[][] buildRandomMatrix(){
		int[][] randomMatrix = new int[3][3];
		
		Random r = new Random();
		
		for(int x = 0; x < randomMatrix.length; x ++) {
			for(int y = 0; y < randomMatrix.length; y ++) {
				int num = r.nextInt(10) + 1;
				randomMatrix[x][y] = num;
			}
		}
	
		return randomMatrix;
		
	}
	
	// Task 3 END
	
	// Task 4 START
	
	public static int[][] buildVectorMatrix(int[] a, int[] b, int[] c){
		int[][] fullMatrix = new int[3][3];
		
		for(int x = 0; x < fullMatrix.length; x ++) {
			for(int y = 0; y < fullMatrix.length; y ++) {
				if(x == 0) {
					fullMatrix[x][y] = a[y];
				}
				else if(x == 1) {
					fullMatrix[x][y] = b[y];
				}
				else {
					fullMatrix[x][y] = c[y];
				}
			}
		}
		
		return fullMatrix;
	}
	
	// Task 4 END
	
	// Task 5 START
	
	public static boolean compareMatrices(int[][] a, int[][] b) {
		boolean same = true;
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				if(a[x][y] != b[x][y]) {
					same = false;
				}
			}
		}	
		
		return same;
	}
	
	// Task 5 END
	
	// Task 6 START
	
	public static int[][] addMatrices(int[][] a, int[][] b){
		int[][] addedMatrix = new int[3][3];
		
		for(int x = 0; x < addedMatrix.length; x ++) {
			for(int y = 0; y < addedMatrix.length; y ++) {
				addedMatrix[x][y] = a[x][y] + b[x][y];
			}
		}
		
		return addedMatrix;
	}
	
	// Task 6 END
	
	// Task 7 START
	
	public static int[][] subtractMatrices(int[][] a, int[][] b){
		int[][] subtractedMatrix = new int[3][3];
		
		for(int x = 0; x < subtractedMatrix.length; x ++) {
			for(int y = 0; y < subtractedMatrix.length; y ++) {
				subtractedMatrix[x][y] = a[x][y] - b[x][y];
			}
		}
		
		return subtractedMatrix;
	}
	
	// Task 7 END
	
	// Task 8 START
	
	public static int[][] scalarProductMatrix(int[][] a, int b){
		int[][] scalarMatrix = new int[3][3];
		
		for(int x = 0; x < scalarMatrix.length; x ++) {
			for(int y = 0; y < scalarMatrix.length; y ++) {
				scalarMatrix[x][y] = a[x][y] * b;
			}
		}
		
		return scalarMatrix;
	}
	
	// Task 8 END
	
	// Task 9 START
	
	public static int rowColumnProduct(int[][] a, int[][] b, int rowIndex, int columnIndex) {
		int product = 0;
		
		for(int x = 0; x < a.length; x++) {
			product += a[rowIndex][x] * b[x][columnIndex];
		}
		
		return product;
	}
	
	public static int[][] multiplyMatrices(int[][] a, int[][] b){
		int[][] newMatrix = new int[3][3];
		
		for(int x = 0; x < newMatrix.length; x ++) {
			for(int y = 0; y < newMatrix.length; y ++) {
				
				newMatrix[x][y] = rowColumnProduct(a, b, x, y);
			}
		}
		
		return newMatrix;
	}
	
	// Task 9 END
	
	// Task 10 START
	
	public static int[][] transposeMatrix(int[][] a){
		int[][] transposedMatrix = new int[a.length][a.length];
		
		for(int x = 0; x < a.length; x ++) {
			for(int y = 0; y < a.length; y ++) {
				
				transposedMatrix[x][y] = a[y][x];
				
			}
		}
		
		return transposedMatrix;
		
	}
	
	// Task 10 END
	
	// Task 11 START
	
	public static boolean isSymmetricMatrix(int[][] a) {
		int[][] aTransposed = transposeMatrix(a);
		
		boolean symmetric;
		
		symmetric = compareMatrices(a, aTransposed);
				
		return symmetric;
	}
	
	// Task 11 END
	
	// Task 12 START
	
	public static int traceMatrix(int[][] a) {
		int trace = 0;
		
		for(int x = 0; x < a.length; x ++) {
			trace += a[x][x];
		}
		
		return trace;
	}
	
	// Task 12 END
	
	// Task 13 START
	
	public static int determinant3x3Matrix(int[][] a) {
		int determinant;
		
		int firstStep = (a[0][0] * a[1][1] * a[2][2]) - (a[0][2] * a[1][1] * a[2][0]);
		// firstStep = (A * E * I) - (C * E * G)
		int secondStep = (a[0][1] * a[1][2] * a[2][0]) - (a[0][1] * a[1][0] * a[2][2]);
		// secondStep = (B * F * G) - (B * D * I)
		int thirdStep = (a[0][2] * a[1][0] * a[2][1]) - (a[0][0] * a[1][2] * a[2][1]);
		// thirdStep = (C * D * H) - (A * F * H)
		
		determinant = firstStep + secondStep + thirdStep;
	
		return determinant;
	}
	
	// Task 13 END
	
	// Task 14 START
	
	public static int[][] powerNMatrix(int[][] a, int power) {
		int[][] newMatrix = a;
		
		for(int x = 0; x < power-1; x ++) {
			newMatrix = multiplyMatrices(newMatrix, a);
		}
		
		return newMatrix;
	}
	
	// Task 14 END

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		// Header
		
		System.out.println("Simone Rittenhouse: Homework 1" + '\n');
		
		// Task 1 Output
		
		System.out.println("Task 1 Output:" + '\n');
		
		int[][] M  = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		displayMatrix(M);
		
		// Task 2 Output
		
		System.out.println('\n' + "Task 2 Output:" + '\n');
		
		int[] V = {11, 2, 13, 4, 15, 6, 17, 8, 19};
		M = buildMatrix(V);
		displayMatrix(M);
		
		// Task 3 Output
		
		System.out.println('\n' + "Task 3 Output:" + '\n');
		
		M = buildRandomMatrix();
		displayMatrix(M);
		
		// Task 4 Output
		
		System.out.println('\n' + "Task 4 Output:" + '\n');
		
		int[] a = {10, 20, 30};
		int[] b = {11, 21, 31};
		int[] c = {12, 22, 32};
		
		M = buildVectorMatrix(a, b, c);
		displayMatrix(M);
		
		// Task 5 Output
		
		System.out.println('\n' + "Task 5 Output:" + '\n');
		
		int[][] m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		int[][] N = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		int[][] P = {{11, 2, 3}, {4, 5, 6}, {7, 8, 19}};
		boolean v = compareMatrices(m, N);
		boolean w = compareMatrices(m, P);
		System.out.println(v);
		System.out.println(w);
		
		// Task 6 Output
		
		System.out.println('\n' + "Task 6 Output:" + '\n');
		
		M = m;
		int[][] n = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
		N = n;
		int[][] MplusN = addMatrices(M, N);
		displayMatrix(MplusN);
		
		// Task 7 Output
		
		System.out.println('\n' + "Task 7 Output:" + '\n');
		
		int[][] NminusM = subtractMatrices(N, M);
		displayMatrix(NminusM);
		
		// Task 8 Output
		
		System.out.println('\n' + "Task 8 Output:" + '\n');
		
		int[][] sM = scalarProductMatrix(M, 10);
		displayMatrix(sM);
		
		// Task 9 Output
		
		System.out.println('\n' + "Task 9 Output:" + '\n');
		
		int[][] newN = {{5, 2, 4}, {4, 6, 6}, {7, 2, 9}};
		N = newN;
		int[][] MN = multiplyMatrices(M, N);
		displayMatrix(MN);
		
		// Task 10 Output
		
		System.out.println('\n' + "Task 10 Output:" + '\n');
		
		int[][] Mt = transposeMatrix(M);
		displayMatrix(M);
		System.out.println();
		displayMatrix(Mt);
		
		// Task 11 Output
		
		System.out.println('\n' + "Task 11 Output:" + '\n');
		
		int[][] A = {{1, 7, 3}, {7, 4, 5}, {3, 5, 6}};
		int[][] B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		boolean sA = isSymmetricMatrix(A);
		System.out.println(sA);
		boolean sB = isSymmetricMatrix(B);
		System.out.println(sB);
		
		// Task 12 Output
		
		System.out.println('\n' + "Task 12 Output:" + '\n');
		
		N = B;
		System.out.println(traceMatrix(N));
		
		// Task 13 Output
		
		System.out.println('\n' + "Task 13 Output:" + '\n');
		
		System.out.println(determinant3x3Matrix(N));
		
		// Task 14 Output
		
		System.out.println('\n' + "Task 14 Output:" + '\n');
		
		displayMatrix(powerNMatrix(N, 3));

	}

}
