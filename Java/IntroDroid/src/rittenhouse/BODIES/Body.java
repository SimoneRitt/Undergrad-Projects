package rittenhouse.BODIES;

public abstract class Body {
	
	private String IDNumber;
	private String BType;
	
	public Body(String IDNumber, String BType) {
		this.IDNumber = IDNumber;
		this.BType = BType;
	}

	public String getIDNumber() {
		return IDNumber;
	}

	public void setIDNumber(String iDNumber) {
		IDNumber = iDNumber;
	}

	public String getBType() {
		return BType;
	}

	public void setBType(String bType) {
		BType = bType;
	}
	
	public abstract void displayInfo();

}
