package org.example.perc;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

public class ObjectToFile {

    public static void saveObjectToFile(Object object, String filePath) {
        ObjectMapper objectMapper = new ObjectMapper();

        try {
            String jsonString = objectMapper.writeValueAsString(object);

            File file = new File(filePath);
            objectMapper.writeValue(file, jsonString);

            System.out.println("Object saved to file: " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static <T> T readObjectFromFile(String filePath, Class<T> valueType) {
        ObjectMapper objectMapper = new ObjectMapper();

        try {
            // Read JSON string from file
            String jsonString = objectMapper.readValue(new File(filePath), String.class);

            // Convert JSON string to object
            return objectMapper.readValue(jsonString, valueType);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}

