package rittenhouse.TIEFIGHTERS;

import rittenhouse.TIEOPS.Bombing;
import rittenhouse.TIESPT.Location;
import rittenhouse.TIESPT.TiePilot;
import rittenhouse.TIEWPNS.LaserCannon;
import rittenhouse.TIEWPNS.ProtonBomb;
import rittenhouse.TIEWPNS.TieWeapon;

public class TieFighter_H extends TieFighter implements Bombing {
	
	// constructor
	
	public TieFighter_H(String IDNumber, TiePilot pilot) {
		super(IDNumber, pilot);
		
		this.setManufacturer("Sienar Fleet Systems");
		this.setModel("Tie Heavy Fighter");
		this.setFighterClass("Close Support");
		this.setLength(7.8);
		this.setWidth(8.6);
		this.setHeight(5.0);
		this.setFuelCapacity(375);
		this.setMaxSpeed(850);
		this.setWpns(new TieWeapon[] {new LaserCannon("L2"), new LaserCannon("L2"), 
				new ProtonBomb("P1"), new ProtonBomb("P1"), new ProtonBomb("P1"), new ProtonBomb("P1"), 
				new ProtonBomb("P1"), new ProtonBomb("P1"), new ProtonBomb("P1"), new ProtonBomb("P1")});
		this.setCurrentLocation(new Location(10, 12, 0));
		this.setLanded(true);
		this.setSpaceborne(false);
		
	}
	
	// methods
	
	@Override
	public void FiresCannons() {
		for(int x = 0; x < 2; x ++) {
			this.getWpns()[x].Fire();
		}
	}
	
	@Override
	public boolean bombTarget() {
		boolean bombs = false;
		
		for(int x = 2; x < this.getWpns().length; x ++) {
			ProtonBomb p = (ProtonBomb) this.getWpns()[x];
			if(p.isDropped() == false) {
				this.getWpns()[x].Fire();
				bombs = true;
				break;
			}
		}
		if(bombs == true) {
			System.out.println("Bombs Away");
			return true;
		}
		else {
			System.out.println("Bombs Expended");
			return false;
		}
	}

}
