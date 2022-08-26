package rittenhouse.BODIES;

public class Cranial extends Body {
	
	private boolean Operational;
	
	public Cranial(String IDNumber) {
		super(IDNumber, "Cranial");
		this.Operational = true;
	}
	

	public Boolean getOperational() {
		return Operational;
	}

	public void setOperational(Boolean operational) {
		Operational = operational;
	}
	
	@Override
	public void displayInfo() {
		System.out.println("Body ID: " + this.getIDNumber() + " Body Type: " + this.getBType());
	}

}