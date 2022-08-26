package rittenhouse.TIEWPNS;

public abstract class TieWeapon {
	
	// variables
	
	private String Model;
	
	// constructor
	
	public TieWeapon(String Model) {
		this.Model = Model;
		
	}
	
	// getters and setters

	public String getModel() {
		return Model;
	}

	public void setModel(String model) {
		Model = model;
	}
	
	// methods
	
	public abstract void Fire();

}
