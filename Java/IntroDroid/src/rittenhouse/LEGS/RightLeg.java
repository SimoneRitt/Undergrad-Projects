package rittenhouse.LEGS;

public class RightLeg extends Leg {
	
	private boolean Operational;
	
	public RightLeg(String IDNumber) {
		super(IDNumber, "Right Leg");
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
		System.out.println("Leg ID: " + this.getIDNumber() + " Leg Type: " + this.getLType());
	}

}
