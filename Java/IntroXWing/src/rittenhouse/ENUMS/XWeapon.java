package rittenhouse.ENUMS;

public enum XWeapon {
	
	KX9("KX9 Laser Cannon",4,"Taim&Bak",18,900),
	KX12("KX12 Laser Cannon",4,"Taim&Bak",21,990),
	KX14("KX14 Laser Cannon",4,"Taim&Bak",23,1250),
	MG7("MG7 Proton Torpedo Launcher",1,"Krupx",7,1100),
	MG7A("MG7A Proton Torpedo Launcher",1,"Krupx",10,1400),
	AX190("AX190 Heavy Laser Cannon",1,"Blas Tech",15,2700);
	
	private String weaponType;
	private int numberOfWeapons;
	private String manufacturer;
	private int maximumRange;
	private int weaponCost;
	
	private XWeapon(String weaponType, int numberOfWeapons, 
			String manufacturer, int maximumRange, int weaponCost) {
		this.weaponType = weaponType;
		this.numberOfWeapons = numberOfWeapons;
		this.manufacturer = manufacturer;
		this.maximumRange = maximumRange;
		this.weaponCost = weaponCost;
	}

	public String getWeaponType() {
		return weaponType;
	}

	public void setWeaponType(String weaponType) {
		this.weaponType = weaponType;
	}

	public int getNumberOfWeapons() {
		return numberOfWeapons;
	}

	public void setNumberOfWeapons(int numberOfWeapons) {
		this.numberOfWeapons = numberOfWeapons;
	}

	public String getManufacturer() {
		return manufacturer;
	}

	public void setManufacturer(String manufacturer) {
		this.manufacturer = manufacturer;
	}

	public int getMaximumRange() {
		return maximumRange;
	}

	public void setMaximumRange(int maximumRange) {
		this.maximumRange = maximumRange;
	}

	public int getWeaponCost() {
		return weaponCost;
	}

	public void setWeaponCost(int weaponCost) {
		this.weaponCost = weaponCost;
	}

}
