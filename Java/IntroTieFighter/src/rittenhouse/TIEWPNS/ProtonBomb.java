package rittenhouse.TIEWPNS;

public class ProtonBomb extends TieWeapon {
	
	// variables 
	
	private int Yield;
	private boolean Dropped;
	
	// constructor
	
	public ProtonBomb(String Model) {
		super(Model);
		
		this.Yield = 7;
		this.Dropped = false;
	}
	
	// getters and setters
	
	public int getYield() {
		return Yield;
	}

	public void setYield(int yield) {
		Yield = yield;
	}

	public boolean isDropped() {
		return Dropped;
	}

	public void setDropped(boolean dropped) {
		Dropped = dropped;
	}
	
	// methods
	
	@Override
	public void Fire() {
		this.Dropped = true;
	}

}
