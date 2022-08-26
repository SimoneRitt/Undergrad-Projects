package rittenhouse.CRYPTO;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class ImperialDecypher_Rittenhouse {
	
	
	// Setting up filename directory paths
	public static String DIRENCRYPT = "/Users/simonerittenhouse/eclipse-workspace/RittenhouseHW3/src/rittenhouse/ENCRYPTED_FILES/";
	public static String DIRPROCESS = "/Users/simonerittenhouse/eclipse-workspace/RittenhouseHW3/src/rittenhouse/PROCESSED_FILES/";

	
	// Getting uppercase and lowercase letters from each line
	public static String[] processLetters(String line) {
		String lowercase = "";
		String uppercase = "";
		
		for(int x = 0; x < line.length(); x ++) {
			
			if(((int)line.charAt(x) >= 97) & ((int)line.charAt(x) <= 122)) {
				lowercase += line.charAt(x);
			}
			
			else if(((int)line.charAt(x) >= 65) & ((int)line.charAt(x) <= 90)) {
				uppercase += line.charAt(x);
			}
			
		}
		
		String[] letters = new String[2];
		
		letters[0] = uppercase;
		letters[1] = lowercase;
		
		return letters;
	}
	
	// Getting digits from each line
	public static String processDigits(String line) {
		String digits = "";
		
		for(int x = 0; x < line.length(); x ++) {
			if(Character.isDigit(line.charAt(x))) {
				digits += line.charAt(x);
			}
		}
		
		return digits;
	}
	
	// Summing the digits in each line
	public static int sumDigitString(String digits) {
		int sum = 0;
		
		for(int x = 0; x < digits.length(); x ++) {
			sum += Character.getNumericValue(digits.charAt(x));
		}
		
		return sum;
	}
	
	// Reading file and writing processed output
	public static void decypher(String encryptedFilename, String processedFilename) throws IOException {
		BufferedReader BR = new BufferedReader(new FileReader(DIRENCRYPT + encryptedFilename));
		BufferedWriter BW = new BufferedWriter(new FileWriter(DIRPROCESS + processedFilename));
		
		String line = "";
		while((line = BR.readLine()) != null) {
			String uppercase = processLetters(line)[0];
			String lowercase = processLetters(line)[1];
			String digits = processDigits(line);
			
			BW.write(uppercase + "->{" + uppercase.length() + "} " 
					+ lowercase + "->{" + lowercase.length() + "} "
					+ digits + "->{" + sumDigitString(digits) + "}");
			BW.write('\n');
		}
		
		BR.close();
		BW.close();
		
	}

}
