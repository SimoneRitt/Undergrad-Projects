package rittenhouse.TIESPT;

public class Location {
	
	// variables
	
	private int X;
	private int Y;
	private int Z;
	
	// constructor
	
	public Location(int X, int Y, int Z) {
		this.X = X;
		this.Y = Y;
		this.Z = Z;
		
	}
	
	// getters and setters
	
	public int getX() {
		return X;
	}

	public void setX(int x) {
		X = x;
	}

	public int getY() {
		return Y;
	}

	public void setY(int y) {
		Y = y;
	}

	public int getZ() {
		return Z;
	}

	public void setZ(int z) {
		Z = z;
	}
	
	// method
	
	public String toString() {
		String location = "Location [X:" + this.X + ", Y:" + this.Y + ", Z:" + this.Z + "]";
		return location;
	}

}
