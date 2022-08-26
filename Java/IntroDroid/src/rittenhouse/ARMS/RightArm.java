package rittenhouse.ARMS;

public class RightArm extends Arm {
	
	private boolean Operational;
	
	public RightArm(String IDNumber) {
		super(IDNumber, "Right Arm");
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
