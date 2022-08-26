package rittenhouse.LEGS;

public abstract class Leg {
	
	private String IDNumber;
	private String LType;
	
	public Leg(String IDNumber, String LType) {
		this.IDNumber = IDNumber;
		this.LType = LType;
	}

	public String getIDNumber() {
		return IDNumber;
	}

	public void setIDNumber(String iDNumber) {
		IDNumber = iDNumber;
	}

	public String getLType() {
		return LType;
	}

	public void setLType(String lType) {
		LType = lType;
	}
	
	public abstract void displayInfo();

}
