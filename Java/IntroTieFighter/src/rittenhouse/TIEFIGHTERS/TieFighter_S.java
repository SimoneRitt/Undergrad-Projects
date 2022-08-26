package rittenhouse.TIEFIGHTERS;

import rittenhouse.TIESPT.Location;
import rittenhouse.TIESPT.TiePilot;
import rittenhouse.TIEWPNS.LaserCannon;
import rittenhouse.TIEWPNS.TieWeapon;

public class TieFighter_S extends TieFighter {
	
	// constructor
	
	public TieFighter_S(String IDNumber, TiePilot pilot) {
		super(IDNumber, pilot);
		
		this.setManufacturer("Sienar Fleet Systems");
		this.setModel("Tie Standard");
		this.setFighterClass("Superiority");
		this.setLength(6.3);
		this.setWidth(6.4);
		this.setHeight(7.5);
		this.setFuelCapacity(200);
		this.setMaxSpeed(1200);
		this.setWpns(new TieWeapon[] {new LaserCannon("L1"), new LaserCannon("L1")});
		this.setCurrentLocation(new Location(10, 12, 0));
		this.setLanded(true);
		this.setSpaceborne(false);
		
	}
	
	// methods
	
	@Override
	public void FiresCannons() {
		for(TieWeapon x: this.getWpns()) {
			x.Fire();
		}
	}

}
