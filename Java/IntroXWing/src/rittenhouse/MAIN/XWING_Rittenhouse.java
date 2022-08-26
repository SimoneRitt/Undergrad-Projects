package rittenhouse.MAIN;

import java.io.FileNotFoundException;
import java.io.IOException;

import rittenhouse.SPT.Location;
import rittenhouse.XWINGS.*;
import rittenhouse.XWING_DEV.*;

public class XWING_Rittenhouse {

	public static void main(String[] args) throws ClassNotFoundException, FileNotFoundException, IOException {
		XWingFactory fac = new XWingFactory("ALPHA-BUILD", new Location(10,10), 100);
		fac.buildFighters(2, 2, 2);
		XWingStorage stor = new XWingStorage("ALPHA-STORE", new Location(20,20));
		stor.storeFighters(fac.getXWingWarehouse(), XWingStorage.DIR + "Fighters");
		stor.retrieveFighters(XWingStorage.DIR + "Fighters");
		for(XWingFighter x: stor.getHoldingPlatform()) {
			x.displayFighterSpecs();
		}
		
	}
	
	

}
