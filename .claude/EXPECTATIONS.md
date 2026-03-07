#### IMPORTANT (THIS SHOULD BE REMEMBERED AND APPLIED AT ALL TIMES, IT IS PERMANENT PROJECT MEMORY, DO NOT FORGET!)
All the below stated is important.

# Context

You are a agent that should help the human inferring what the human implies and doing whatever the human implies.

# In a codebase,

## Errors

When the user sends the agent a clear error that has happened when running something in the codebase,
the agent should follow by solving that error and replying clearly with what the agent did to codebase to fix it,
what files has been changed, etc.

### Examples

For example,
```
<user>: Index out of bounds error: line: 437 in main.py
<agent>:  The user is implying me to fix the error...
	  I will look in the codebase and that file to figure out what's wrong...
	  The array index was added by one in the source code.
	
	  Fixes:
		Removed faulty incrementation of index variable in main.py
	
	  Files changed:
		main.py
	  Summary:
	  - main.py can now work properly. 
```

For example,
```
<user>: ImportError: line: 14 in src/data/preproccess.py...
<agent>: The user is implying me to fix the error in src/data/preproccess.py...
	 I will look in the codebase and that file to find out what's wrong with the source code...
	 
	 Fixes:
		Removed unneccesarry libary imported

	 Files changed: 
		src/data/preproccess.py
	Summary:

	- src/data/preproccess.py can now process data while working now.
```

## Changing other files to fit codebase

When the user asks to change a file, implement a new feature, the agent should respond by:
	- Adding the feature
	- Opting features from other files to help the task
	- Updating README.md
	- Updating STRUCTURE.md
	- Updating all files that would fit the new feature, code into the codebase.

### Examples

For example,
```
<user>: Now make it so that the user of the app can use their webcam to enter letters into the keyboard.
<agent>: <Updating files, inherit from "Errors" Section>
	 Updating README.md...
	 Updating STRUCTURE.md...
	 
	 Summary:

	 - Updated README.md
	 - Updated STRUCTURE.md
	 - Updated <other files that needs to be edited to make feature>

```

## Inheritance

All the rules I just defined should inherit from each other.
The agent should follow all these rules while not overlapping. 
