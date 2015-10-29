local TransposeX, parent = torch.class('nn.TransposeX', 'nn.Module')

function TransposeX:__init()
   parent.__init(self)
end

function TransposeX:updateOutput(input)
  self.output = {}
  for i, k in ipairs(input) do
    local channel = k:size()[1]
    local height  = k:size()[2]
    local width   = k:size()[3]
    assert( height == 1, 'height must be 1')
    table.insert(self.output, k:resize(channel,width):t())
  end
  return self.output
end

function TransposeX:updateGradInput(input, gradOutput)
   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end
