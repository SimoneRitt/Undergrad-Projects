package rittenhouse.ENUMS;

public enum XEngine {
	
	CPG04Z("Novaldex","Cryogenic Power Generator",3700,8000),
	FTE5L5("Incom-FreiTek","Fusial Thrust Engine",3800,8750),
	FTE5L9("Incom-FreiTek","Fusial Thrust Engine",3925,9050);
	
	private String manufacturer;
	private String engineType;
	private int powerOutput;
	private int engineCost;
	
	private XEngine(String manufacturer, String engineType, 
			int powerOutput, int engineCost) {
		this.manufacturer = manufacturer;
		this.engineType = engineType;
		this.powerOutput = powerOutput;
		this.engineCost = engineCost;
	}

	public String getManufacturer() {
		return manufacturer;
	}

	public void setManufacturer(String manufacturer) {
		this.manufacturer = manufacturer;
	}

	public String getEngineType() {
		return engineType;
	}

	public void setEngineType(String engineType) {
		this.engineType = engineType;
	}

	public int getPowerOutput() {
		return powerOutput;
	}

	public void setPowerOutput(int powerOutput) {
		this.powerOutput = powerOutput;
	}

	public int getEngineCost() {
		return engineCost;
	}

	public void setEngineCost(int engineCost) {
		this.engineCost = engineCost;
	}
	
	

}
