package rittenhouse.LEGS;

public class LeftLeg extends Leg {
	
	private boolean Operational;
	
	public LeftLeg(String IDNumber) {
		super(IDNumber, "Left Leg");
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
