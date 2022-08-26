package rittenhouse.XWING_DEV;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayDeque;
import java.util.Iterator;

import rittenhouse.SPT.*;
import rittenhouse.XWINGS.*;

public class XWingStorage {
	
	private String name;
	private Location storageLoc;
	private ArrayDeque<XWingFighter> holdingPlatform;
	
	public static final String DIR = "/Users/simonerittenhouse/eclipse-workspace/Rittenhouse_HW7/src/rittenhouse/XWING_DEV/";

	public XWingStorage(String name, Location storageLoc) {
		this.name = name;
		this.storageLoc = storageLoc;
		this.holdingPlatform = new ArrayDeque<>();
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public Location getStorageLoc() {
		return storageLoc;
	}

	public void setStorageLoc(Location storageLoc) {
		this.storageLoc = storageLoc;
	}

	public ArrayDeque<XWingFighter> getHoldingPlatform() {
		return holdingPlatform;
	}

	public void setHoldingPlatform(ArrayDeque<XWingFighter> holdingPlatform) {
		this.holdingPlatform = holdingPlatform;
	}

	public boolean storeFighters(ArrayDeque<XWingFighter> wh, String fn) {
		try {
			ObjectOutputStream OS = new ObjectOutputStream(new FileOutputStream(fn));
			Iterator<XWingFighter> it = wh.iterator();
			while(it.hasNext()) {
				OS.writeObject(it.next());
			}
			OS.close();
		}catch(FileNotFoundException e) {
			e.printStackTrace();
			return false;
		}catch(IOException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	public boolean retrieveFighters(String fn) throws FileNotFoundException, IOException {
		ObjectInputStream OS = new ObjectInputStream(new FileInputStream(fn));
		XWingFighter X;
		
		try {
			while((X = (XWingFighter) OS.readObject()) != null) {
				this.holdingPlatform.add(X);
			}
			OS.close();
		}catch (EOFException e) {
			OS.close();
		}catch (IOException | ClassNotFoundException e) {
			OS.close();
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
}
