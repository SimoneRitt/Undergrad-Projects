package rittenhouse.ARMS;

public class LeftArm extends Arm {
	
	private boolean Operational;
	
	public LeftArm(String IDNumber) {
		super(IDNumber, "Left Arm");
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
		System.out.println("Arm ID: " + this.getIDNumber() + " Arm Type: " + this.getAType());
	}

}
