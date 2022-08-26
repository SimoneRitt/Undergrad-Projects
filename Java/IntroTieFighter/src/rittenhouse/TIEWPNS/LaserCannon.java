package rittenhouse.TIEWPNS;

public class LaserCannon extends TieWeapon {
	
	// variables
	
	private int MaxRange;
	private int AmmoCapacity;
	private int CurrentAmmoCount;
	
	// constructor
	
	public LaserCannon(String Model) {
		super(Model);
		
		this.MaxRange = 1000;
		this.AmmoCapacity = 500;
		this.CurrentAmmoCount = 500;
	}
	
	// getters and setters
	
	public int getMaxRange() {
		return MaxRange;
	}

	public void setMaxRange(int maxRange) {
		MaxRange = maxRange;
	}

	public int getAmmoCapacity() {
		return AmmoCapacity;
	}

	public void setAmmoCapacity(int ammoCapacity) {
		AmmoCapacity = ammoCapacity;
	}

	public int getCurrentAmmoCount() {
		return CurrentAmmoCount;
	}

	public void setCurrentAmmoCount(int currentAmmoCount) {
		CurrentAmmoCount = currentAmmoCount;
	}
	
	// methods
	
	@Override
	public void Fire() {
		if(this.CurrentAmmoCount > 0) {
			this.CurrentAmmoCount -= 1;
			System.out.println("Firing Cannon");
		}
		else {
			System.out.println("Ammunition Expended");
		}
	}

}
