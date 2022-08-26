package rittenhouse.PARTICLE;

public class Location { // EMAIL PROF
	
	// variables

	private int X;
	private int Y;
	
	// constructor
	
	public Location(int x, int y) {
		this.X = x;
		this.Y = y;
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
	
	// methods
	
	public double euclideanDistance(Location L) {
		double distance = Math.pow((Math.pow((L.getX()-this.X), 2) + Math.pow((L.getY()-this.Y), 2)), 0.5);
		return distance;
	}
	
	public String toString() {
		String L = "(" + this.X + "," + this.Y + ")";
		return L;
	}
	
}
