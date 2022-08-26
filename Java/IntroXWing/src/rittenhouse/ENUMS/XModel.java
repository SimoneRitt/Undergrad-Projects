package rittenhouse.ENUMS;

public enum XModel {
	
	T65B("Starfighter","Incom","Rebel Alliance",150000,"YR-1"),
	T70("Starfighter","Incom-FreiTek","Rebel Alliance",200000,"YR-5"),
	T85("Starfighter","Incom-FreiTek","Rebel Alliance",220000,"YR-25");
	
	private String role;
	private String manufacturer;
	private String nationality;
	private int modelCost;
	private String yearInService;
	
	private XModel(String role, String manufacturer, 
			String nationality, int modelCost, String yearInService) {
		this.role = role;
		this.manufacturer = manufacturer;
		this.nationality = nationality;
		this.modelCost = modelCost;
		this.yearInService = yearInService;
	}

	public String getRole() {
		return role;
	}

	public void setRole(String role) {
		this.role = role;
	}

	public String getManufacturer() {
		return manufacturer;
	}

	public void setManufacturer(String manufacturer) {
		this.manufacturer = manufacturer;
	}

	public String getNationality() {
		return nationality;
	}

	public void setNationality(String nationality) {
		this.nationality = nationality;
	}

	public int getModelCost() {
		return modelCost;
	}

	public void setModelCost(int modelCost) {
		this.modelCost = modelCost;
	}

	public String getYearInService() {
		return yearInService;
	}

	public void setYearInService(String yearInService) {
		this.yearInService = yearInService;
	}
	
	
}
