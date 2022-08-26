package rittenhouse.BODIES;

public class Torso extends Body {
	
	private boolean Operational;
	
	public Torso(String IDNumber) {
		super(IDNumber, "Torso");
		this.Operational = true;
	}
	
	
	public boolean getOperational() {
		return Operational;
	}

	public void setOperational(boolean operational) {
		Operational = operational;
	}

	@Override
	public void displayInfo() {
		System.out.println("Body ID: " + this.getIDNumber() + " Body Type: " + this.getBType());
	}

}
